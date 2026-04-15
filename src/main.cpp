// src/main.cu
#include "gpt2_engine.cuh"
#include "gpt2_types.cuh"
#include "tokenizer.h"
#include "sampler.cuh"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <fstream>
#include <sstream>

using namespace ftxui;

struct AppState {
    std::mutex mtx;
    std::string generated_text;
    InferenceMetrics metrics;
    std::atomic<bool> running{false};
    std::atomic<bool> quit{false};
};

static std::string fmt(float v, int prec = 1) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

static Element MetricsPanel(const InferenceMetrics& m) {
    auto section = [](std::string title, std::vector<Element> children) {
        return vbox({
            text(title) | bold,
            separator(),
            vbox(std::move(children)),
        });
    };

    float vram_frac = m.vram_total_mb > 0 ? m.vram_used_mb / m.vram_total_mb : 0.0f;

    return vbox({
        section("Latency", {
            hbox({text("TTFT       "), filler(), text(fmt(m.ttft_ms) + " ms")}),
            hbox({text("TPOT       "), filler(), text(fmt(m.tpot_ms) + " ms")}),
            hbox({text("ITL P95    "), filler(), text(fmt(m.itl_p95_ms) + " ms")}),
            hbox({text("ITL min/max"), filler(),
                  text(fmt(m.itl_min_ms) + "/" + fmt(m.itl_max_ms) + " ms")}),
            hbox({text("TPS        "), filler(), text(fmt(m.tps))}),
            hbox({text("Tokens     "), filler(),
                  text(std::to_string(m.tokens_generated) + "/" +
                       std::to_string(m.max_tokens))}),
        }),
        text(""),
        section("Resource", {
            hbox({text("KV "), gauge(m.kv_cache_pct / 100.0f) | flex,
                  text(" " + fmt(m.kv_cache_pct, 0) + "%")}),
            hbox({text("VRAM "), gauge(vram_frac) | flex,
                  text(" " + fmt(m.vram_used_mb, 0) + "/" +
                       fmt(m.vram_total_mb, 0) + " MB")}),
            hbox({text("MBU  "), filler(), text(fmt(m.mbu * 100.0f, 1) + " %")}),
        }),
        text(""),
        hbox({text("End-to-end "), filler(),
              text(m.end_to_end_ms > 0 ? fmt(m.end_to_end_ms) + " ms" : "--")}),
    }) | border | size(WIDTH, EQUAL, 42);
}

// ============================================================
// --compare mode: 3-column TUI racing PyTorch FP32 vs cuBLAS FP32 vs INT8.
// Sequential (no GPU contention); fair timings.
// ============================================================
namespace compare {

struct PaneState {
    std::string name;
    std::string text;
    InferenceMetrics metrics;
    int status = 0;  // 0 pending, 1 running, 2 done
};

struct State {
    std::mutex mtx;
    std::array<PaneState, 3> panes;
    std::atomic<bool> running{false};
};

static float parse_json_num(const std::string& s, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) return 0.0f;
    p = s.find(':', p);
    if (p == std::string::npos) return 0.0f;
    return (float)atof(s.c_str() + p + 1);
}

static Element pane_element(const PaneState& p, Color header_color) {
    std::string status_str = p.status == 0 ? "pending"
                           : p.status == 1 ? "running" : "done";
    auto header = hbox({
        text(p.name) | bold | color(header_color),
        filler(),
        text("[" + status_str + "]") | dim,
    });

    auto body_text = p.text.empty() ? text("...") | dim : paragraph(p.text);

    auto fmt_v = [](float v) {
        char buf[32]; snprintf(buf, sizeof(buf), "%.1f", v); return std::string(buf);
    };
    const auto& m = p.metrics;
    auto metrics_box = vbox({
        hbox({text("TTFT "), filler(), text(fmt_v(m.ttft_ms) + " ms")}),
        hbox({text("TPOT "), filler(), text(fmt_v(m.tpot_ms) + " ms")}),
        hbox({text("TPS  "), filler(), text(fmt_v(m.tps))}),
        hbox({text("VRAM "), filler(),
              text(fmt_v(m.vram_used_mb) + " MB")}),
    });

    return vbox({
        header,
        separator(),
        body_text | flex,
        separator(),
        metrics_box,
    }) | border | flex;
}

static Element speedup_bar(const State& st) {
    float pt_tps = st.panes[0].metrics.tps;
    if (pt_tps <= 0.0f) return text(" awaiting PyTorch baseline... ") | dim | center;
    auto fmt_x = [](float v) {
        char buf[32]; snprintf(buf, sizeof(buf), "%.2fx", v); return std::string(buf);
    };
    Elements items;
    items.push_back(text(" Speedup vs PyTorch: ") | bold);
    for (int i = 0; i < 3; ++i) {
        float s = st.panes[i].metrics.tps / pt_tps;
        items.push_back(text(st.panes[i].name + " ") | dim);
        Color c = Color::Default;
        if (i != 0) c = (s >= 1.0f) ? Color::Green : Color::Red;
        items.push_back(text(fmt_x(s)) | bold | color(c));
        if (i < 2) items.push_back(text("   "));
    }
    return hbox(std::move(items)) | border;
}

static void run_pytorch(const std::string& prompt, int max_tokens,
                        State& st, ScreenInteractive& screen) {
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        st.panes[0].status = 1;
    }
    screen.Post(Event::Custom);

    // Escape quotes in prompt for shell.
    std::string esc;
    for (char c : prompt) {
        if (c == '"' || c == '\\') esc += '\\';
        esc += c;
    }
    std::string cmd = "uv run python python/benchmark_pytorch.py "
                      "--prompt \"" + esc + "\" "
                      "--max-tokens " + std::to_string(max_tokens) +
                      " --stream-protocol 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    std::string metrics_line;
    if (pipe) {
        char buf[8192];
        while (fgets(buf, sizeof(buf), pipe)) {
            std::string line(buf);
            if (!line.empty() && line.back() == '\n') line.pop_back();
            if (line.rfind("TOK ", 0) == 0) {
                std::string tok = line.substr(4);
                size_t pos = 0;
                while ((pos = tok.find("\\n", pos)) != std::string::npos) {
                    tok.replace(pos, 2, "\n"); pos++;
                }
                std::lock_guard<std::mutex> lk(st.mtx);
                st.panes[0].text += tok;
                screen.Post(Event::Custom);
            } else if (line.rfind("MET ", 0) == 0) {
                metrics_line = line.substr(4);
            }
        }
        pclose(pipe);
    }

    std::lock_guard<std::mutex> lk(st.mtx);
    auto& m = st.panes[0].metrics;
    m.ttft_ms = parse_json_num(metrics_line, "ttft_ms");
    m.tpot_ms = parse_json_num(metrics_line, "tpot_ms");
    m.tps = parse_json_num(metrics_line, "tps");
    m.itl_p95_ms = parse_json_num(metrics_line, "itl_p95_ms");
    m.end_to_end_ms = parse_json_num(metrics_line, "end_to_end_ms");
    m.vram_used_mb = parse_json_num(metrics_line, "vram_mb");
    m.tokens_generated = (int)parse_json_num(metrics_line, "tokens");
    st.panes[0].status = 2;
    screen.Post(Event::Custom);
}

static void run_slick(int idx, const std::string& model_dir,
                      InferenceBackend backend,
                      const std::string& prompt, int max_tokens,
                      State& st, ScreenInteractive& screen) {
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        st.panes[idx].status = 1;
    }
    screen.Post(Event::Custom);

    GPT2Engine engine(model_dir, backend);
    BPETokenizer tokenizer(model_dir);
    auto tokens = tokenizer.encode(prompt);
    InferenceMetrics m;
    engine.generate(tokens, max_tokens, 0.8f, 40, /*greedy=*/false,
        [&](int id) {
            std::lock_guard<std::mutex> lk(st.mtx);
            st.panes[idx].text += tokenizer.decode(id);
            screen.Post(Event::Custom);
        }, m);

    std::lock_guard<std::mutex> lk(st.mtx);
    st.panes[idx].metrics = m;
    st.panes[idx].status = 2;
    screen.Post(Event::Custom);
}

}  // namespace compare

static int run_compare_tui(const std::string& model_dir,
                           const std::string& initial_prompt,
                           int max_tokens) {
    using namespace compare;

    State state;
    state.panes[0] = {"PyTorch FP32", "", {}, 0};
    state.panes[1] = {"SLICK cuBLAS", "", {}, 0};
    state.panes[2] = {"SLICK INT8",   "", {}, 0};

    auto screen = ScreenInteractive::Fullscreen();
    std::string input_text = initial_prompt;
    auto input = Input(&input_text, "Prompt...");

    auto run_sequence = [&](const std::string& prompt) {
        {
            std::lock_guard<std::mutex> lk(state.mtx);
            for (auto& p : state.panes) {
                p.text.clear();
                p.metrics = {};
                p.status = 0;
            }
        }
        run_pytorch(prompt, max_tokens, state, screen);
        run_slick(1, model_dir, InferenceBackend::CUBLAS_FP32,
                  prompt, max_tokens, state, screen);
        run_slick(2, model_dir, InferenceBackend::SLICK_INT8,
                  prompt, max_tokens, state, screen);
        state.running = false;
        screen.Post(Event::Custom);
    };

    auto component = CatchEvent(input, [&](Event event) -> bool {
        if (event == Event::Return && !input_text.empty() && !state.running) {
            std::string prompt = input_text;
            state.running = true;
            std::thread([&, prompt]() { run_sequence(prompt); }).detach();
            return true;
        }
        if (event == Event::Escape) {
            screen.Exit();
            return true;
        }
        return false;
    });

    auto renderer = Renderer(component, [&] {
        std::lock_guard<std::mutex> lk(state.mtx);
        auto p0 = pane_element(state.panes[0], Color::Yellow);
        auto p1 = pane_element(state.panes[1], Color::Cyan);
        auto p2 = pane_element(state.panes[2], Color::Green);

        auto main_area = hbox({p0, p1, p2}) | flex;
        auto speedup = speedup_bar(state);

        bool any_done = state.panes[0].status == 2 || state.panes[1].status == 2
                        || state.panes[2].status == 2;
        auto input_bar = hbox({
            text("> "),
            component->Render() | flex,
            text(state.running ? " [racing...]" : (any_done ? " [done — Enter to re-run]" : "")) | dim,
        }) | border;

        return vbox({
            text(" SLICK — 3-Way Inference Race ") | bold | center,
            main_area,
            speedup,
            input_bar,
        });
    });

    screen.Loop(renderer);
    return 0;
}

int main(int argc, char** argv) {
    std::string model_dir = "models/gpt2-int8";
    std::string initial_prompt;
    int max_tokens = 256;
    float temperature = 0.8f;
    int top_k_val = 50;
    bool greedy = false;
    bool bench_mode = false;
    bool compare_mode = false;
    InferenceBackend backend = InferenceBackend::SLICK_INT8;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) initial_prompt = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k_val = atoi(argv[++i]);
        else if (strcmp(argv[i], "--greedy") == 0) greedy = true;
        else if (strcmp(argv[i], "--bench") == 0) bench_mode = true;
        else if (strcmp(argv[i], "--compare") == 0) compare_mode = true;
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            ++i;
            if (strcmp(argv[i], "cublas") == 0) backend = InferenceBackend::CUBLAS_FP32;
            else if (strcmp(argv[i], "int8") == 0) backend = InferenceBackend::SLICK_INT8;
        }
    }

    if (compare_mode) {
        return run_compare_tui(model_dir, initial_prompt.empty()
                                ? std::string("The meaning of life is")
                                : initial_prompt,
                               max_tokens);
    }

    const char* backend_name = (backend == InferenceBackend::CUBLAS_FP32)
                                ? "cublas_fp32" : "slick_int8";
    printf("Loading model from %s (backend=%s)...\n", model_dir.c_str(), backend_name);
    GPT2Engine engine(model_dir, backend);
    BPETokenizer tokenizer(model_dir);
    printf("Model loaded. Vocab: %d\n", tokenizer.vocab_size());

    if (bench_mode) {
        if (initial_prompt.empty()) initial_prompt = "The meaning of life is";
        auto tokens = tokenizer.encode(initial_prompt);
        printf("Prompt (%d tokens): %s", (int)tokens.size(), initial_prompt.c_str());
        fflush(stdout);

        InferenceMetrics metrics;
        engine.generate(tokens, max_tokens, temperature, top_k_val, greedy,
                        [&](int id) {
                            printf("%s", tokenizer.decode(id).c_str());
                            fflush(stdout);
                        }, metrics);

        printf("\n\n==================================================\n");
        printf("SLICK %s\n", backend_name);
        printf("==================================================\n");
        printf("Tokens:       %d\n", metrics.tokens_generated);
        printf("TTFT:         %.1f ms\n", metrics.ttft_ms);
        printf("TPOT:         %.1f ms\n", metrics.tpot_ms);
        printf("TPS:          %.1f\n", metrics.tps);
        printf("ITL P95:      %.1f ms\n", metrics.itl_p95_ms);
        printf("ITL min/max:  %.1f/%.1f ms\n", metrics.itl_min_ms, metrics.itl_max_ms);
        printf("End-to-end:   %.1f ms\n", metrics.end_to_end_ms);
        printf("KV Cache:     %.0f%%\n", metrics.kv_cache_pct);
        printf("VRAM:         %.0f/%.0f MB\n", metrics.vram_used_mb, metrics.vram_total_mb);
        printf("MBU:          %.1f%%\n", metrics.mbu * 100.0f);

        char json_file[256];
        snprintf(json_file, sizeof(json_file), "%s_results.json", backend_name);
        FILE* jf = fopen(json_file, "w");
        if (jf) {
            fprintf(jf, "{\n");
            fprintf(jf, "  \"backend\": \"%s\",\n", backend_name);
            fprintf(jf, "  \"tokens\": %d,\n", metrics.tokens_generated);
            fprintf(jf, "  \"ttft_ms\": %.1f,\n", metrics.ttft_ms);
            fprintf(jf, "  \"tpot_ms\": %.1f,\n", metrics.tpot_ms);
            fprintf(jf, "  \"tps\": %.1f,\n", metrics.tps);
            fprintf(jf, "  \"itl_p95_ms\": %.1f,\n", metrics.itl_p95_ms);
            fprintf(jf, "  \"end_to_end_ms\": %.1f,\n", metrics.end_to_end_ms);
            fprintf(jf, "  \"vram_mb\": %.0f\n", metrics.vram_used_mb);
            fprintf(jf, "}\n");
            fclose(jf);
            printf("\nResults saved to %s\n", json_file);
        }
        return 0;
    }

    // --- Interactive TUI mode ---
    AppState state;
    auto screen = ScreenInteractive::Fullscreen();

    std::string input_text;
    auto input = Input(&input_text, "Enter prompt...");

    auto component = CatchEvent(input, [&](Event event) -> bool {
        if (event == Event::Return && !input_text.empty() && !state.running) {
            std::string prompt = input_text;
            input_text.clear();

            state.running = true;
            {
                std::lock_guard<std::mutex> lock(state.mtx);
                state.generated_text = prompt;
                state.metrics = {};
                state.metrics.max_tokens = max_tokens;
            }

            std::thread([&, prompt]() {
                auto tokens = tokenizer.encode(prompt);
                engine.generate(tokens, max_tokens, temperature, top_k_val, greedy,
                    [&](int id) {
                        std::string decoded = tokenizer.decode(id);
                        {
                            std::lock_guard<std::mutex> lock(state.mtx);
                            state.generated_text += decoded;
                        }
                        screen.Post(Event::Custom);
                    }, state.metrics);

                state.running = false;
                screen.Post(Event::Custom);
            }).detach();

            return true;
        }
        if (event == Event::Escape) {
            state.quit = true;
            screen.Exit();
            return true;
        }
        return false;
    });

    auto renderer = Renderer(component, [&] {
        std::lock_guard<std::mutex> lock(state.mtx);

        auto output_panel = vbox({
            text("Output") | bold,
            separator(),
            paragraph(state.generated_text) | flex,
        }) | border | flex;

        auto metrics_panel = MetricsPanel(state.metrics);

        auto main_area = hbox({
            output_panel,
            metrics_panel,
        }) | flex;

        auto input_bar = hbox({
            text("> "),
            component->Render() | flex,
            text(state.running ? " [generating...]" : "") | dim,
        }) | border;

        return vbox({
            text(" SLICK GPT-2 Engine ") | bold | center,
            main_area,
            input_bar,
        });
    });

    screen.Loop(renderer);
    return 0;
}
