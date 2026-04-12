// include/tokenizer.h
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class BPETokenizer {
public:
    explicit BPETokenizer(const std::string& model_dir);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& token_ids) const;

    int vocab_size() const { return (int)id_to_token_.size(); }
    int eos_token() const { return 50256; }

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int> merge_rank_;

    std::vector<std::string> bpe(const std::string& word) const;
};
