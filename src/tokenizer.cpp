// src/tokenizer.cpp
#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>

// GPT-2 byte-level BPE: map each of 256 bytes to a printable unicode codepoint.
// This mirrors HuggingFace's bytes_to_unicode().
static std::unordered_map<unsigned char, std::string> build_byte_encoder() {
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    std::vector<int> cs = bs;

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    std::unordered_map<unsigned char, std::string> be;
    for (size_t i = 0; i < bs.size(); ++i) {
        int cp = cs[i];
        std::string s;
        if (cp < 0x80) {
            s += (char)cp;
        } else if (cp < 0x800) {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        } else {
            s += (char)(0xE0 | (cp >> 12));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        }
        be[(unsigned char)bs[i]] = s;
    }
    return be;
}

static std::unordered_map<std::string, unsigned char> build_byte_decoder() {
    auto be = build_byte_encoder();
    std::unordered_map<std::string, unsigned char> bd;
    for (auto& kv : be) bd[kv.second] = kv.first;
    return bd;
}

BPETokenizer::BPETokenizer(const std::string& model_dir) {
    std::ifstream vf(model_dir + "/vocab.json");
    if (!vf) { fprintf(stderr, "Failed to open vocab.json\n"); exit(1); }
    std::string vcontent((std::istreambuf_iterator<char>(vf)),
                          std::istreambuf_iterator<char>());

    id_to_token_.resize(50257);
    size_t pos = 1;
    while (pos < vcontent.size()) {
        auto ks = vcontent.find('"', pos);
        if (ks == std::string::npos) break;
        auto ke = vcontent.find('"', ks + 1);
        while (ke != std::string::npos && vcontent[ke - 1] == '\\') {
            ke = vcontent.find('"', ke + 1);
        }
        std::string key = vcontent.substr(ks + 1, ke - ks - 1);
        std::string clean;
        for (size_t i = 0; i < key.size(); ++i) {
            if (key[i] == '\\' && i + 1 < key.size()) {
                if (key[i+1] == '"')      { clean += '"';  ++i; }
                else if (key[i+1] == '\\') { clean += '\\'; ++i; }
                else if (key[i+1] == 'n')  { clean += '\n'; ++i; }
                else if (key[i+1] == 't')  { clean += '\t'; ++i; }
                else if (key[i+1] == 'u') {
                    unsigned int cp = 0;
                    for (int j = 0; j < 4 && i + 2 + j < key.size(); ++j) {
                        char c = key[i + 2 + j];
                        cp <<= 4;
                        if (c >= '0' && c <= '9') cp |= c - '0';
                        else if (c >= 'a' && c <= 'f') cp |= c - 'a' + 10;
                        else if (c >= 'A' && c <= 'F') cp |= c - 'A' + 10;
                    }
                    if (cp < 0x80) { clean += (char)cp; }
                    else if (cp < 0x800) {
                        clean += (char)(0xC0 | (cp >> 6));
                        clean += (char)(0x80 | (cp & 0x3F));
                    } else {
                        clean += (char)(0xE0 | (cp >> 12));
                        clean += (char)(0x80 | ((cp >> 6) & 0x3F));
                        clean += (char)(0x80 | (cp & 0x3F));
                    }
                    i += 5;
                }
                else { clean += key[i]; }
            } else {
                clean += key[i];
            }
        }

        auto cs = vcontent.find(':', ke);
        auto vs = vcontent.find_first_of("0123456789", cs);
        auto ve = vcontent.find_first_not_of("0123456789", vs);
        int id = atoi(vcontent.substr(vs, ve - vs).c_str());

        token_to_id_[clean] = id;
        if (id >= 0 && id < (int)id_to_token_.size()) id_to_token_[id] = clean;

        pos = ve;
        auto next = vcontent.find_first_of(",}", pos);
        if (next == std::string::npos || vcontent[next] == '}') break;
        pos = next + 1;
    }

    std::ifstream mf(model_dir + "/merges.txt");
    if (!mf) { fprintf(stderr, "Failed to open merges.txt\n"); exit(1); }
    std::string line;
    int rank = 0;
    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string a = line.substr(0, sp);
        std::string b = line.substr(sp + 1);
        merges_.push_back({a, b});
        merge_rank_[a + " " + b] = rank++;
    }
}

std::vector<std::string> BPETokenizer::bpe(const std::string& word) const {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < word.size()) {
        int len = 1;
        unsigned char c = (unsigned char)word[i];
        if (c >= 0xC0 && c < 0xE0)       len = 2;
        else if (c >= 0xE0 && c < 0xF0)  len = 3;
        else if (c >= 0xF0)              len = 4;
        tokens.push_back(word.substr(i, len));
        i += len;
    }

    if (tokens.size() <= 1) return tokens;

    while (true) {
        int best_rank = INT_MAX;
        int best_pos = -1;
        for (int j = 0; j < (int)tokens.size() - 1; ++j) {
            std::string key = tokens[j] + " " + tokens[j + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = j;
            }
        }
        if (best_pos < 0) break;

        std::string merged = tokens[best_pos] + tokens[best_pos + 1];
        tokens[best_pos] = merged;
        tokens.erase(tokens.begin() + best_pos + 1);
    }
    return tokens;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    auto be = build_byte_encoder();
    std::vector<int> ids;

    std::istringstream iss(text);
    std::string word;
    bool first = true;
    while (iss >> word) {
        std::string encoded;
        if (!first) encoded += be[0x20];
        for (unsigned char c : word) encoded += be[c];
        first = false;

        auto bpe_tokens = bpe(encoded);
        for (auto& t : bpe_tokens) {
            auto it = token_to_id_.find(t);
            if (it != token_to_id_.end()) ids.push_back(it->second);
        }
    }
    return ids;
}

std::string BPETokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= (int)id_to_token_.size()) return "";
    auto bd = build_byte_decoder();
    const std::string& token = id_to_token_[token_id];

    std::string result;
    size_t i = 0;
    while (i < token.size()) {
        int len = 1;
        unsigned char c = (unsigned char)token[i];
        if (c >= 0xC0 && c < 0xE0)       len = 2;
        else if (c >= 0xE0 && c < 0xF0)  len = 3;
        else if (c >= 0xF0)              len = 4;
        std::string ch = token.substr(i, len);
        auto it = bd.find(ch);
        if (it != bd.end()) result += (char)it->second;
        else result += ch;
        i += len;
    }
    return result;
}

std::string BPETokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (int id : token_ids) result += decode(id);
    return result;
}
