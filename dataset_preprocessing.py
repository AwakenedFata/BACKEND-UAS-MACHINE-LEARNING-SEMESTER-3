import json
import pandas as pd
from typing import List, Dict
from text_preprocessing import IndonesianTextPreprocessor

class DatasetPreprocessor:
    
    def __init__(self, dataset_path: str = None):
        self.preprocessor = IndonesianTextPreprocessor()
        self.dataset = None
        self.processed_data = []
        
        if dataset_path:
            self.load_dataset(dataset_path)
    
    def load_dataset(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
    
    def preprocess_dataset(self, 
                          stem: bool = True,
                          remove_stopwords: bool = False) -> list:
        processed = []
        
        if not self.dataset:
            return processed
        
        for intent in self.dataset['intents']:
            tag = intent['tag']
            
            for pattern in intent['patterns']:
                original = pattern
                
                preprocessed = self.preprocessor.preprocess(
                    pattern,
                    lowercase=True,
                    remove_punct=True,
                    normalize_slang=True,
                    stem=stem,
                    remove_stopwords=remove_stopwords,
                    keep_numbers=True
                )
                
                processed.append({
                    'original': original,
                    'preprocessed': preprocessed,
                    'intent': tag
                })
        
        self.processed_data = processed
        
        return processed
    
    def save_preprocessed(self, output_path: str):
        if not self.processed_data:
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
    
    def get_dataframe(self) -> pd.DataFrame:
        if not self.processed_data:
            return None
        
        df = pd.DataFrame(self.processed_data)
        return df
    
    def get_statistics(self):
        if not self.processed_data:
            return
        
        df = self.get_dataframe()
        
        return {
            'total_samples': len(df),
            'total_intents': df['intent'].nunique(),
            'intent_distribution': df['intent'].value_counts().to_dict()
        }
    
    def balance_dataset(self, method='oversample', min_samples=10):
        if not self.processed_data:
            return
        
        df = self.get_dataframe()
        
        if method == 'oversample':
            intent_counts = df['intent'].value_counts()
            max_count = intent_counts.max()
            
            balanced_data = []
            for intent in df['intent'].unique():
                intent_data = df[df['intent'] == intent].to_dict('records')
                
                while len(intent_data) < max_count:
                    intent_data.extend(intent_data[:max_count - len(intent_data)])
                
                balanced_data.extend(intent_data[:max_count])
            
            self.processed_data = balanced_data
        
        elif method == 'undersample':
            intent_counts = df['intent'].value_counts()
            min_count = max(intent_counts.min(), min_samples)
            
            balanced_data = []
            for intent in df['intent'].unique():
                intent_data = df[df['intent'] == intent].to_dict('records')
                balanced_data.extend(intent_data[:min_count])
            
            self.processed_data = balanced_data
    
    def augment_data(self):
        if not self.processed_data:
            return
        
        augmented = []
        
        for item in self.processed_data:
            augmented.append(item)
            
            if len(item['preprocessed'].split()) > 3:
                no_stopwords = self.preprocessor.preprocess(
                    item['original'],
                    remove_stopwords=True
                )
                
                if no_stopwords != item['preprocessed']:
                    augmented.append({
                        'original': item['original'] + " (augmented)",
                        'preprocessed': no_stopwords,
                        'intent': item['intent']
                    })
        
        self.processed_data = augmented


class DatasetAugmenter:
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        self.common_typos = {
            'halo': ['hallo', 'haloo', 'halloo', 'haloooo', 'hlo'],
            'hai': ['haii', 'haiii', 'hay', 'hei', 'heii'],
            'apa': ['apaa', 'apaaa', 'apa sih', 'apaan'],
            'gimana': ['gmn', 'gmna', 'gimanaa', 'gimanaaa'],
            'berapa': ['brp', 'brapa', 'berapaa', 'berapaan'],
            'terima kasih': ['makasih', 'makasi', 'makasii', 'thx', 'thanks', 'tq', 'tengkyu'],
            'tidak': ['ga', 'gak', 'gk', 'nggak', 'ngga', 'kagak', 'enggak'],
            'sudah': ['udah', 'udh', 'dah', 'uda', 'sdh'],
            'belum': ['blm', 'blom', 'belom', 'blum'],
            'bisa': ['bs', 'bsa', 'bisaa', 'bisaaa'],
            'mau': ['mw', 'mo', 'mauu', 'mauuu', 'pengen', 'pgn'],
            'ada': ['adaa', 'adaaa', 'ad'],
            'kaos': ['kaoss', 'kaus', 'baju', 't-shirt', 'tshirt'],
            'harga': ['hrg', 'hrgnya', 'harganya', 'brp harga'],
            'order': ['pesen', 'pesan', 'psn', 'oderr'],
            'assalamualaikum': ['aslmkm', 'asalamualaikum', 'assalamu alaikum', 'asslmkm'],
        }
    
    def generate_repeated_char_variants(self, word: str, max_repeats: int = 4) -> List[str]:
        variants = [word]
        
        vowels = 'aeiou'
        for i, char in enumerate(word.lower()):
            if char in vowels:
                for repeat in range(2, max_repeats + 1):
                    new_word = word[:i] + char * repeat + word[i+1:]
                    variants.append(new_word)
        
        if word and word[-1].lower() in vowels:
            for repeat in range(2, max_repeats + 1):
                variants.append(word + word[-1] * (repeat - 1))
        
        return list(set(variants))
    
    def generate_typo_variants(self, word: str) -> List[str]:
        word_lower = word.lower()
        if word_lower in self.common_typos:
            return self.common_typos[word_lower]
        return []
    
    def augment_pattern(self, pattern: str) -> List[str]:
        augmented = [pattern]
        
        words = pattern.split()
        for i, word in enumerate(words):
            repeated_variants = self.generate_repeated_char_variants(word)
            for variant in repeated_variants[:3]:
                if variant != word:
                    new_pattern = ' '.join(words[:i] + [variant] + words[i+1:])
                    augmented.append(new_pattern)
            
            typo_variants = self.generate_typo_variants(word)
            for variant in typo_variants[:2]:
                new_pattern = ' '.join(words[:i] + [variant] + words[i+1:])
                augmented.append(new_pattern)
        
        return list(set(augmented))
    
    def augment_dataset(self, target_intents: List[str] = None) -> Dict:
        augmented_dataset = {
            'metadata': self.dataset.get('metadata', {}),
            'intents': []
        }
        
        priority_intents = [
            'greeting', 'goodbye', 'thank_you', 'casual_talk',
            'produk_kaos', 'harga_kaos', 'cara_order', 'ukuran_size'
        ]
        
        if target_intents is None:
            target_intents = priority_intents
        
        for intent in self.dataset.get('intents', []):
            tag = intent['tag']
            patterns = intent['patterns']
            responses = intent['responses']
            
            new_patterns = list(patterns)
            
            if tag in target_intents:
                for pattern in patterns[:20]:
                    augmented = self.augment_pattern(pattern)
                    new_patterns.extend(augmented)
            
            new_patterns = list(set(new_patterns))
            
            augmented_dataset['intents'].append({
                'tag': tag,
                'patterns': new_patterns,
                'responses': responses
            })
        
        total_new = sum(len(i['patterns']) for i in augmented_dataset['intents'])
        total_old = sum(len(i['patterns']) for i in self.dataset['intents'])
        
        augmented_dataset['metadata']['total_patterns'] = f"{total_new}+"
        augmented_dataset['metadata']['augmented'] = True
        augmented_dataset['metadata']['original_patterns'] = total_old
        
        return augmented_dataset
    
    def save_augmented(self, output_path: str, target_intents: List[str] = None):
        augmented = self.augment_dataset(target_intents)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(augmented, f, ensure_ascii=False, indent=4)
        
        return augmented


if __name__ == '__main__':
    import sys
    
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'datasets.json'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'datasets.json'
    
    augmenter = DatasetAugmenter(input_path)
    result = augmenter.save_augmented(output_path)
    
    print(f"Augmented dataset saved to {output_path}")
    print(f"Original patterns: {result['metadata'].get('original_patterns', 'N/A')}")
    print(f"New patterns: {result['metadata'].get('total_patterns', 'N/A')}")
