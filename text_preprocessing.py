import re
import string
import json
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class IndonesianTextPreprocessor:
   
    def __init__(self):
        self.stemmer_factory = StemmerFactory()
        self.stemmer = self.stemmer_factory.create_stemmer()
        
        self.stopword_factory = StopWordRemoverFactory()
        self.stopwords = set(self.stopword_factory.get_stop_words())
        
        self.custom_stopwords = {
            'ya', 'iya', 'yah', 'sih', 'dong', 'deh', 'nih', 'lho', 
            'kan', 'kah', 'lah', 'tah', 'pun'
        }
        self.stopwords.update(self.custom_stopwords)
        
        self.slang_dict = {
            'hai': 'halo', 'hey': 'halo', 'hy': 'halo', 'p': 'halo',
            'helo': 'halo', 'hallo': 'halo', 'haloo': 'halo', 'hola': 'halo',
            'hi': 'halo', 'hay': 'halo', 'hei': 'halo', 'heii': 'halo',
            'yo': 'halo', 'yoo': 'halo', 'yow': 'halo',
            
            'pagi': 'selamat pagi', 'pgi': 'selamat pagi',
            'siang': 'selamat siang', 
            'sore': 'selamat sore',
            'malam': 'selamat malam', 'mlm': 'selamat malam', 'malem': 'selamat malam',
            
            'assalamualaikum': 'salam', 'assalamu': 'salam', 'assalam': 'salam',
            'asalamualaikum': 'salam', 'aslmkm': 'salam', 'asslmkm': 'salam',
            'assalamualaikumwarahmatullahiwabarakatuh': 'salam',
            'waalaikumsalam': 'salam balas', 'walaikumsalam': 'salam balas',
            
            'gmn': 'bagaimana', 'gimana': 'bagaimana', 'gmna': 'bagaimana',
            'gimanaa': 'bagaimana', 'gmana': 'bagaimana',
            'brp': 'berapa', 'berpa': 'berapa', 'brapa': 'berapa', 'berapaa': 'berapa',
            'knp': 'kenapa', 'knapa': 'kenapa',
            'dmn': 'dimana', 'dmana': 'dimana', 'dimanaa': 'dimana', 'mana': 'dimana',
            'kmn': 'kemana', 'kmana': 'kemana',
            'kpn': 'kapan', 'kapann': 'kapan',
            'bgmn': 'bagaimana', 'bgmna': 'bagaimana',
            'ap': 'apa', 'apaa': 'apa', 'apaan': 'apa',
            'sapa': 'siapa', 'siapaa': 'siapa',
            
            'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'nggak': 'tidak',
            'ngga': 'tidak', 'g': 'tidak', 'kagak': 'tidak', 'kaga': 'tidak',
            'gaa': 'tidak', 'gaak': 'tidak', 'gakkk': 'tidak', 'enggak': 'tidak',
            'ndak': 'tidak', 'tak': 'tidak', 'tdk': 'tidak',
            
            'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah', 'sdh': 'sudah',
            'uda': 'sudah', 'udahh': 'sudah',
            'blm': 'belum', 'blom': 'belum', 'belom': 'belum', 'blum': 'belum',
            
            'org': 'orang', 'orng': 'orang', 'ornag': 'orang',
            'jg': 'juga', 'jga': 'juga', 'jugaa': 'juga',
            'dgn': 'dengan', 'dg': 'dengan', 'ama': 'dengan', 'sama': 'dengan',
            'sm': 'sama', 'ma': 'sama',
            'tp': 'tapi', 'tpi': 'tapi', 'tapii': 'tapi',
            'krn': 'karena', 'krna': 'karena', 'soalnya': 'karena', 'karna': 'karena',
            'utk': 'untuk', 'buat': 'untuk', 'bt': 'untuk', 'untk': 'untuk',
            'dpt': 'dapat', 'dapet': 'dapat', 'dpet': 'dapat',
            'pake': 'pakai', 'pk': 'pakai', 'pakee': 'pakai', 'pkai': 'pakai',
            'byk': 'banyak', 'bnyk': 'banyak', 'bnyak': 'banyak',
            'emg': 'memang', 'emang': 'memang', 'mmg': 'memang',
            'sbg': 'sebagai', 'sbagai': 'sebagai',
            'trs': 'terus', 'trus': 'terus', 'trss': 'terus',
            'bs': 'bisa', 'bsa': 'bisa', 'bisaa': 'bisa',
            'lg': 'lagi', 'lgi': 'lagi', 'lagii': 'lagi',
            'yg': 'yang', 'yng': 'yang',
            'klo': 'kalau', 'kalo': 'kalau', 'klw': 'kalau', 'klu': 'kalau',
            'sy': 'saya', 'aku': 'saya', 'gw': 'saya', 'gue': 'saya', 'gua': 'saya',
            'km': 'kamu', 'kmu': 'kamu', 'lu': 'kamu', 'lo': 'kamu', 'elu': 'kamu',
            'ni': 'ini', 'nih': 'ini',
            'tu': 'itu', 'tuh': 'itu',
            
            'merch': 'merchandise', 'merchandis': 'merchandise',
            'ganci': 'gantungan kunci', 'keychain': 'gantungan kunci',
            'merhcandise': 'merchandise', 'merchandice': 'merchandise', 'merchan': 'merchandise',
            'totebag': 'tas', 'tote': 'tas',
            'stiker': 'sticker',
            'cap': 'topi',
            
            'tf': 'transfer', 
            'cod': 'bayar ditempat', 'cash': 'tunai',
            'disc': 'diskon', 'discount': 'diskon', 'potongan': 'diskon', 'pot': 'diskon',
            'ongkir': 'ongkos kirim', 'ongkos': 'ongkos kirim',
            
            'cepet': 'cepat', 'cpet': 'cepat', 'cpat': 'cepat',
            'mahal': 'mahal', 'expensive': 'mahal', 'mahall': 'mahal',
            'murah': 'murah', 'cheap': 'murah', 'murahh': 'murah',
            'bagus': 'bagus', 'oke': 'baik', 'ok': 'baik', 'okey': 'baik', 'okay': 'baik',
            
            'gede': 'besar', 'gedhe': 'besar', 'gde': 'besar',
            'kecil': 'kecil', 'kecl': 'kecil', 'kcil': 'kecil',
            'jumbo': 'besar',
            
            'min': 'admin', 'bang': 'admin', 'kak': 'admin',
            'bro': 'admin', 'sis': 'admin', 'gan': 'admin', 'minn': 'admin',
            
            'thx': 'terima kasih', 'thanks': 'terima kasih', 'thank': 'terima kasih',
            'tq': 'terima kasih', 'makasi': 'terima kasih', 'mksh': 'terima kasih',
            'makasih': 'terima kasih', 'tengkyu': 'terima kasih', 'tenkyu': 'terima kasih',
            'trims': 'terima kasih', 'trim': 'terima kasih', 'terimakasih': 'terima kasih',
            
            'bye': 'sampai jumpa', 'dadah': 'sampai jumpa', 'byee': 'sampai jumpa',
            'seeyou': 'sampai jumpa', 'see': 'sampai jumpa',
            
            'mw': 'mau', 'mo': 'mau', 'mauu': 'mau', 'pengen': 'mau', 'pgn': 'mau',
            'pingin': 'mau', 'pengin': 'mau',
            
            'ada': 'ada', 'adaa': 'ada',
            'jual': 'jual', 'juall': 'jual',
            'beli': 'beli', 'blii': 'beli',
            'order': 'pesan', 'pesen': 'pesan', 'psn': 'pesan',
            'hrg': 'harga', 'hrgnya': 'harga', 'harganya': 'harga',
            'brpaan': 'berapa', 'berapaan': 'berapa',
            
            'kaos': 'kaos', 'kaoss': 'kaos', 'kaus': 'kaos',
            'baju': 'kaos', 'bajuu': 'kaos',
            'tshirt': 'kaos', 't-shirt': 'kaos',
            
            'warna': 'warna', 'warnaa': 'warna', 'wrn': 'warna',
            'size': 'ukuran', 'ukuran': 'ukuran', 'sz': 'ukuran',
            
            'kirim': 'kirim', 'krim': 'kirim', 'krm': 'kirim',
            'sampai': 'sampai', 'sampe': 'sampai', 'smpe': 'sampai',
            
            'lama': 'lama', 'lamaa': 'lama',
            'cpt': 'cepat', 'cpet': 'cepat',
            
            'rekomen': 'rekomendasi', 'recommend': 'rekomendasi', 'rekomendasiin': 'rekomendasi',
            'saran': 'rekomendasi',
        }
        
    def normalize_repeated_chars(self, text: str) -> str:
        result = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        common_words = {
            'halo': ['haloo', 'halloo', 'haloooo', 'halooo'],
            'hai': ['haii', 'haiii', 'haiiii', 'haiiiii'],
            'hi': ['hii', 'hiii', 'hiiii'],
            'oke': ['okee', 'okeee', 'okeeee'],
            'ya': ['yaa', 'yaaa', 'yaaaa'],
            'iya': ['iyaa', 'iyaaa', 'iyaaaa', 'y', 'yy', 'yyy', 'ye', 'yee', 'yow', 'yoww', 'yowww', 'iyu', 'iyuu', 'iyuuu', 'iyow', 'iyoww', 'iyo', 'iyoo'],
            'terima kasih': ['makasii', 'makasiii', 'makasiiii', 'mksh', 'mksi', 'mksih', 'mks'],
            'apa': ['apaa', 'apaaa', 'apaaaa'],
            'mau': ['mauu', 'mauuu', 'mauuuu', 'maww', 'maw', 'mo', 'mao'],
            'bisa': ['bisaa', 'bisaaa', 'bisaaaa'],
            'ada': ['adaa', 'adaaa', 'adaaaa'],
            'tidak': ['gaa', 'gk', 'g', 'ga', 'gaaa', 'gaaaa', 'gakk', 'gakkk', 'nggk', 'ngak', 'nggak', 'engga', 'enggak', 'engak'],
        }
        
        words = result.split()
        normalized = []
        for word in words:
            found = False
            for base, variants in common_words.items():
                if word.lower() in variants or word.lower() == base:
                    normalized.append(base)
                    found = True
                    break
            if not found:
                normalized.append(word)
                
        return ' '.join(normalized)
        
    def normalize_slang(self, text: str) -> str:
        words = text.split()
        normalized = []
        
        for word in words:
            if word.lower() in self.slang_dict:
                normalized.append(self.slang_dict[word.lower()])
            else:
                normalized.append(word)
        
        return ' '.join(normalized)
    
    def remove_punctuation(self, text: str) -> str:
        translator = str.maketrans('', '', string.punctuation.replace('?', ''))
        return text.translate(translator)
    
    def remove_extra_whitespace(self, text: str) -> str:
        return ' '.join(text.split())
    
    def remove_numbers(self, text: str, keep_numbers: bool = False) -> str:
        if keep_numbers:
            return text
        return re.sub(r'\d+', '', text)
    
    def stem_text(self, text: str) -> str:
        return self.stemmer.stem(text)
    
    def remove_stopwords(self, text: str, keep_stopwords: bool = False) -> str:
        if keep_stopwords:
            return text
        
        words = text.split()
        filtered = [w for w in words if w.lower() not in self.stopwords]
        return ' '.join(filtered)
    
    def preprocess(self, 
                   text: str, 
                   lowercase: bool = True,
                   remove_punct: bool = True,
                   normalize_slang: bool = True,
                   stem: bool = True,
                   remove_stopwords: bool = True,
                   keep_numbers: bool = False) -> str:
        
        if lowercase:
            text = text.lower()
        
        text = self.remove_extra_whitespace(text)
        
        text = self.normalize_repeated_chars(text)
        
        if normalize_slang:
            text = self.normalize_slang(text)
        
        if remove_punct:
            text = self.remove_punctuation(text)
        
        text = self.remove_numbers(text, keep_numbers=keep_numbers)
        
        text = self.remove_extra_whitespace(text)
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        if stem:
            text = self.stem_text(text)
        
        text = self.remove_extra_whitespace(text)
        
        return text.strip()


class FuzzyMatcher:
    
    def __init__(self, dataset_path: str = None):
        self.patterns = []
        self.pattern_to_intent = {}
        
        if dataset_path:
            self.load_patterns(dataset_path)
    
    def load_patterns(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for intent in data.get('intents', []):
            tag = intent['tag']
            for pattern in intent['patterns']:
                pattern_lower = pattern.lower().strip()
                self.patterns.append(pattern_lower)
                self.pattern_to_intent[pattern_lower] = tag
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def normalized_levenshtein(self, s1: str, s2: str) -> float:
        distance = self.levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1.0 - (distance / max_len)
    
    def find_best_match(self, text: str, threshold: float = 0.6) -> Optional[Tuple[str, str, float]]:
        text_lower = text.lower().strip()
        
        if text_lower in self.pattern_to_intent:
            return (text_lower, self.pattern_to_intent[text_lower], 1.0)
        
        best_pattern = None
        best_intent = None
        best_score = 0.0
        
        for pattern in self.patterns:
            score = self.normalized_levenshtein(text_lower, pattern)
            
            if score > best_score:
                best_score = score
                best_pattern = pattern
                best_intent = self.pattern_to_intent[pattern]
        
        if best_score >= threshold:
            return (best_pattern, best_intent, best_score)
        
        return None
    
    def find_matches(self, text: str, top_n: int = 3, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        text_lower = text.lower().strip()
        
        scores = []
        for pattern in self.patterns:
            score = self.normalized_levenshtein(text_lower, pattern)
            if score >= threshold:
                scores.append((pattern, self.pattern_to_intent[pattern], score))
        
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_n]
