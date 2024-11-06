from collections import defaultdict
import argparse

class BPETrain(): 
    def __init__(self, corpus, vocab_size, outpath, vocab): 
        self.corpus = corpus 
        self.vocab_size = vocab_size
        self.outpath = outpath
        self.vocab = vocab # list of ordered merge operations
        self.word_freq = defaultdict(int)
        self.splits = {}
        self.pair_freqs = defaultdict(int)

    def train(self):
        f = open(self.vocab, "w", encoding="utf8")
        tokenized_text = open(self.outpath, "w", encoding="utf8")

        for i, line in enumerate(self.corpus): 
            wordList = line.rstrip("\n").split(" ")
            for word in wordList: 
                self.word_freq[word] += 1
            self.corpus[i] = wordList

        # split each word into its characters 
        self.splits = {word: [c for c in word] for word in self.word_freq.keys()}
        for word in self.splits.keys():
            self.splits[word].append("_")
        
        # merge
        numMerges = 0
        self.compute_pair_freqs()
        while numMerges < self.vocab_size: 

            # find the most frequent pair, w/ tie breaking
            most_freq_pair = ()
            max_freq = 0
            for pair, freq in self.pair_freqs.items(): 
                if freq > max_freq: 
                    most_freq_pair = pair
                    max_freq = freq
                elif freq == max_freq: # tie break 
                    if pair[1] < most_freq_pair[1] or \
                    (pair[1] == most_freq_pair[1] and pair[0] < most_freq_pair[0]): 
                        most_freq_pair = pair
                        max_freq = freq
            
            if most_freq_pair == (): # no more pairs left 
                break
            f.write(most_freq_pair[0] + " " + most_freq_pair[1])
            f.write("\n")
            self.merge(*most_freq_pair)
            numMerges += 1
        
        # write the BPE tokenized file
        for i, wordList in enumerate(self.corpus):
            lineStr = ""
            for word in wordList: 
                tokenized = self.splits[word]
                for token in tokenized: 
                    for ch in token:
                        lineStr += ch if ch != "_" else ""
                    if token != "_": lineStr += " "

            tokenized_text.write(lineStr.rstrip(" "))
            tokenized_text.write("\n")

        f.close()
        tokenized_text.close()

    def compute_pair_freqs(self): 
        for word, freq in self.word_freq.items(): 
            charList = self.splits[word]
            for i in range(len(charList) - 1):
                pair = (charList[i], charList[i+1])
                self.pair_freqs[pair] += freq

    def merge(self, first, second): 
        for word, charList in self.splits.items():
            i = 0
            while i < len(charList) - 1: 
                if first == charList[i] and second == charList[i+1]:
                    if i != 0: 
                        self.pair_freqs[(charList[i-1], charList[i])] -= self.word_freq[word]
                        if self.pair_freqs[(charList[i-1], charList[i])] == 0: del self.pair_freqs[(charList[i-1], charList[i])]
                        self.pair_freqs[(charList[i-1], first+second)] += self.word_freq[word]
                    if i != len(charList) - 2:
                        self.pair_freqs[(charList[i+1], charList[i+2])] -= self.word_freq[word]
                        if self.pair_freqs[(charList[i+1], charList[i+2])] == 0: del self.pair_freqs[(charList[i+1], charList[i+2])]
                        self.pair_freqs[(first+second, charList[i+2])] += self.word_freq[word]

                    self.pair_freqs[(first, second)] -= self.word_freq[word]
                    if self.pair_freqs[(first, second)] == 0: del self.pair_freqs[(first, second)]

                    charList = charList[:i] + [first+second] + charList[i+2:]

                else: i += 1
                self.splits[word] = charList
            


def BPETest(inpath, outpath, vocab): 
    tokenized_test_file = open(outpath, "w", encoding="utf8")
    learned_vocab = open(vocab, encoding="utf8") 
    test_file = open(inpath, encoding="utf8")
    test_text = test_file.readlines() # list of lines in test_file
    for i, line in enumerate(test_text): 
        wordList = line.rstrip("\n").split(" ") 
        tokenizedList = []
        for word in wordList: 
            tokenizedList.extend([l for l in word])
            tokenizedList.append("_")
        test_text[i] = tokenizedList
        # ["Hello World"] --> ["Hello", "World"] 
        # --> ["H", "e", "l", "l", "o", "_", "W", "o", "r", "l", "d", "_"]
    while True: 
        merged = learned_vocab.readline().rstrip("\n")
        if not merged: break 
        first, second = merged.split(" ")
        for i, line in enumerate(test_text):
            j = 0
            while j < len(line)-1: 
                if line[j] == first and line[j+1] == second: 
                    line = line[:j] + [first+second] + line[j+2:]
                else: j += 1
                test_text[i] = line
                
    # write back to tokenized_test_file
    for line in test_text: 
        lineStr = ""
        for token in line: 
            for ch in token:
                lineStr += ch if ch != "_" else ""
            if token != "_": lineStr += " "

        tokenized_test_file.write(lineStr.rstrip(" "))
        tokenized_test_file.write("\n")
                            
    learned_vocab.close()
    test_file.close()
    tokenized_test_file.close()


 # Create the top-level parser
parser = argparse.ArgumentParser(description="BPE processing script")

# Mutually exclusive group to enforce either --learn_bpe or --apply_bpe
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--learn_bpe", action="store_true", help="Learn BPE from input text")
group.add_argument("--apply_bpe", action="store_true", help="Apply BPE to input text")

parser.add_argument("--inpath", required=True, help="Path to the input text file")
parser.add_argument("--outpath", required=True, help="Path to the output text file")
parser.add_argument("--vocab", required=True, help="Path to the vocab file")
parser.add_argument("--vocab_size", default = 10000, type=int, help="Size of the vocabulary (required for learn_bpe)")

args = parser.parse_args()

if args.learn_bpe:
    with open(args.inpath, encoding="utf8") as f:
        corpus = f.readlines()
        bpe = BPETrain(corpus, args.vocab_size, args.outpath, args.vocab)
        bpe.train()
elif args.apply_bpe:
    BPETest(args.inpath, args.outpath, args.vocab)




