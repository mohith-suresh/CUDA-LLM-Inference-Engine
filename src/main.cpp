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
#include <algorithm>

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

int main(int argc, char** argv) {
    std::string model_dir = "models/gpt2-int8";
    std::string initial_prompt;
    int max_tokens = 256;
    float temperature = 0.8f;
    int top_k_val = 50;
    bool greedy = false;
    bool bench_mode = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) initial_prompt = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k_val = atoi(argv[++i]);
        else if (strcmp(argv[i], "--greedy") == 0) greedy = true;
        else if (strcmp(argv[i], "--bench") == 0) bench_mode = true;
    }

    printf("Loading model from %s...\n", model_dir.c_str());
    GPT2Engine engine(model_dir);
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

        printf("\n\n--- Metrics ---\n");
        printf("TTFT:        %.1f ms\n", metrics.ttft_ms);
        printf("TPOT:        %.1f ms\n", metrics.tpot_ms);
        printf("ITL P95:     %.1f ms\n", metrics.itl_p95_ms);
        printf("TPS:         %.1f\n", metrics.tps);
        printf("Tokens:      %d\n", metrics.tokens_generated);
        printf("KV Cache:    %.0f%%\n", metrics.kv_cache_pct);
        printf("VRAM:        %.0f / %.0f MB\n", metrics.vram_used_mb, metrics.vram_total_mb);
        printf("End-to-end:  %.1f ms\n", metrics.end_to_end_ms);
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
