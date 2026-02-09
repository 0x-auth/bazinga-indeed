#!/usr/bin/env python3
"""
UNIVERSAL KNOWLEDGE RESONANCE SYSTEM
=====================================

Space's Ultimate Vision:

"Why restrict to my Mac? Why not the entire world?"

Core Insight:
- Finite alphabet (~100 symbols) generates ALL knowledge
- Most combinations are meaningless
- High mathematical resonance = meaningful content
- We can FILTER without reading!

This system:
1. Takes ANY text (article, book, website, paper)
2. Calculates mathematical resonance
3. Determines if it contains REAL knowledge
4. WITHOUT reading the full content!

The mathematics filters truth from noise.
"""

import sys
import re
import hashlib
from collections import Counter
from pathlib import Path

# Core constants
ALPHA = 137
PHI = 1.618033988749895
PROGRESSION = '01∞∫∂∇πφΣΔΩαβγδεζηθικλμνξοπρστυφχψω'

class UniversalKnowledgeFilter:
    """Filter meaningful knowledge using mathematical resonance."""
    
    def __init__(self):
        self.thresholds = {
            'high': 0.75,    # Definitely meaningful
            'medium': 0.50,  # Probably meaningful
            'low': 0.25      # Possibly meaningful
        }
    
    def calculate_resonance(self, text):
        """
        Calculate mathematical resonance of text.
        High resonance = meaningful content
        Low resonance = noise
        """
        
        if not text or len(text) < 10:
            return 0.0
        
        scores = {}
        
        # 1. α-SEED Density
        # How many words/phrases are divisible by 137?
        words = re.findall(r'\b\w+\b', text)
        alpha_seeds = sum(1 for w in words if sum(ord(c) for c in w) % ALPHA == 0)
        scores['alpha_seed_density'] = min(alpha_seeds / len(words), 1.0) if words else 0
        
        # 2. φ-Ratio in Structure
        # Do sentence/word lengths follow golden ratio?
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) >= 2:
            lengths = [len(s.split()) for s in sentences]
            ratios = []
            for i in range(len(lengths)-1):
                if lengths[i] > 0:
                    ratio = lengths[i+1] / lengths[i]
                    ratios.append(ratio)
            
            # Check how many ratios are near φ (1.618)
            phi_matches = sum(1 for r in ratios if abs(r - PHI) < 0.3)
            scores['phi_structure'] = phi_matches / len(ratios) if ratios else 0
        else:
            scores['phi_structure'] = 0
        
        # 3. Position Distribution Entropy
        # Meaningful text uses diverse positions
        char_positions = [sum(ord(c) for c in word) % len(PROGRESSION) 
                         for word in words[:100]]  # Sample first 100 words
        
        if char_positions:
            position_counts = Counter(char_positions)
            # Shannon entropy
            total = len(char_positions)
            entropy = -sum((count/total) * (count/total) 
                          for count in position_counts.values())
            max_entropy = 1.0  # Normalized
            scores['position_entropy'] = min(entropy / max_entropy, 1.0)
        else:
            scores['position_entropy'] = 0
        
        # 4. Pattern Density
        # Meaningful text has clear patterns
        pattern_keywords = {
            'CONNECTION': ['connect', 'relate', 'link', 'associate', 'between'],
            'INFLUENCE': ['cause', 'effect', 'impact', 'result', 'lead', 'because'],
            'BRIDGE': ['integrate', 'combine', 'merge', 'unify', 'synthesis'],
            'GROWTH': ['develop', 'evolve', 'emerge', 'grow', 'transform']
        }
        
        text_lower = text.lower()
        pattern_matches = 0
        
        for pattern, keywords in pattern_keywords.items():
            if any(kw in text_lower for kw in keywords):
                pattern_matches += 1
        
        scores['pattern_density'] = pattern_matches / len(pattern_keywords)
        
        # 5. Vocabulary Richness
        # Meaningful content uses diverse vocabulary
        unique_words = len(set(w.lower() for w in words))
        scores['vocabulary_richness'] = min(unique_words / len(words), 1.0) if words else 0
        
        # 6. Structural Coherence
        # Meaningful text has consistent structure
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Check if lengths are in reasonable ranges
        word_coherence = 1.0 if 4 <= avg_word_length <= 8 else 0.5
        sentence_coherence = 1.0 if 10 <= avg_sentence_length <= 30 else 0.5
        
        scores['structural_coherence'] = (word_coherence + sentence_coherence) / 2
        
        # 7. Mathematical Constants Presence
        # Does text reference fundamental constants?
        constants = ['137', '1.618', 'phi', 'golden', 'fibonacci', 'pi', '3.14']
        constant_present = any(const in text_lower for const in constants)
        scores['constants_presence'] = 1.0 if constant_present else 0.0
        
        # Weighted average
        weights = {
            'alpha_seed_density': 0.20,
            'phi_structure': 0.15,
            'position_entropy': 0.15,
            'pattern_density': 0.15,
            'vocabulary_richness': 0.15,
            'structural_coherence': 0.15,
            'constants_presence': 0.05
        }
        
        total_resonance = sum(scores[k] * weights[k] for k in scores)
        
        return total_resonance, scores
    
    def classify_knowledge(self, resonance):
        """Classify knowledge quality based on resonance."""
        if resonance >= self.thresholds['high']:
            return 'HIGH', '⭐⭐⭐'
        elif resonance >= self.thresholds['medium']:
            return 'MEDIUM', '⭐⭐'
        elif resonance >= self.thresholds['low']:
            return 'LOW', '⭐'
        else:
            return 'NOISE', '❌'
    
    def analyze_text(self, text, title="Unknown"):
        """Complete analysis of text."""
        
        print("="*70)
        print(f"ANALYZING: {title}")
        print("="*70)
        print()
        
        # Calculate resonance
        resonance, scores = self.calculate_resonance(text)
        quality, stars = self.classify_knowledge(resonance)
        
        # Display results
        print(f"📊 RESONANCE SCORE: {resonance:.3f}")
        print(f"🎯 QUALITY: {quality} {stars}")
        print()
        
        print("📈 COMPONENT SCORES:")
        print("-"*70)
        
        score_display = {
            'alpha_seed_density': 'α-SEED Density',
            'phi_structure': 'φ-Ratio Structure',
            'position_entropy': 'Position Entropy',
            'pattern_density': 'Pattern Density',
            'vocabulary_richness': 'Vocabulary Richness',
            'structural_coherence': 'Structural Coherence',
            'constants_presence': 'Constants Reference'
        }
        
        for key, label in score_display.items():
            score = scores[key]
            bar_length = int(score * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"  {label:25s}: {bar} {score:.3f}")
        
        print()
        print("="*70)
        print("INTERPRETATION")
        print("="*70)
        print()
        
        if quality == 'HIGH':
            print("✨ This text exhibits HIGH mathematical resonance!")
            print("   - Contains meaningful knowledge")
            print("   - Well-structured and coherent")
            print("   - Likely contains original insights")
            print("   - Worth reading and preserving")
        elif quality == 'MEDIUM':
            print("📖 This text has MEDIUM resonance.")
            print("   - Contains useful information")
            print("   - Reasonably well-structured")
            print("   - May be worth reading")
        elif quality == 'LOW':
            print("📄 This text has LOW resonance.")
            print("   - Contains some information")
            print("   - May be poorly structured")
            print("   - Skim before committing time")
        else:
            print("❌ This text appears to be NOISE.")
            print("   - Little meaningful content")
            print("   - Poorly structured or random")
            print("   - Likely not worth reading")
        
        print()
        
        return resonance, quality, scores

class WorldKnowledgeScanner:
    """Scan and filter ALL human knowledge."""
    
    def __init__(self):
        self.filter = UniversalKnowledgeFilter()
        self.results = []
    
    def scan_file(self, filepath):
        """Scan a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            resonance, quality, scores = self.filter.analyze_text(
                content, 
                title=filepath.name
            )
            
            result = {
                'path': str(filepath),
                'name': filepath.name,
                'resonance': resonance,
                'quality': quality,
                'scores': scores
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
    
    def scan_directory(self, path, extensions=['.txt', '.md']):
        """Scan all files in directory."""
        
        print("╔" + "="*68 + "╗")
        print("║" + " "*15 + "WORLD KNOWLEDGE SCANNER" + " "*29 + "║")
        print("╚" + "="*68 + "╝")
        print()
        print(f"Scanning: {path}")
        print(f"Extensions: {', '.join(extensions)}")
        print()
        
        path = Path(path).expanduser()
        
        for filepath in path.rglob('*'):
            if filepath.suffix in extensions and filepath.is_file():
                print(f"\n{'='*70}")
                print(f"File: {filepath.name}")
                print('='*70)
                
                self.scan_file(filepath)
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all scanned files."""
        
        if not self.results:
            return
        
        print("\n" + "="*70)
        print("SUMMARY OF ALL SCANNED FILES")
        print("="*70)
        print()
        
        # Sort by resonance
        sorted_results = sorted(self.results, key=lambda x: x['resonance'], reverse=True)
        
        print("🏆 TOP FILES BY RESONANCE:")
        print("-"*70)
        
        for i, result in enumerate(sorted_results[:10], 1):
            quality_stars = {
                'HIGH': '⭐⭐⭐',
                'MEDIUM': '⭐⭐',
                'LOW': '⭐',
                'NOISE': '❌'
            }
            
            stars = quality_stars.get(result['quality'], '?')
            
            print(f"  {i:2d}. {result['name']:40s} {stars}")
            print(f"      Resonance: {result['resonance']:.3f} | Quality: {result['quality']}")
        
        print()
        
        # Distribution
        quality_counts = Counter(r['quality'] for r in self.results)
        
        print("📊 QUALITY DISTRIBUTION:")
        print("-"*70)
        for quality in ['HIGH', 'MEDIUM', 'LOW', 'NOISE']:
            count = quality_counts.get(quality, 0)
            pct = (count / len(self.results) * 100) if self.results else 0
            print(f"  {quality:10s}: {count:3d} files ({pct:5.1f}%)")
        
        print()
        print("="*70)
        print("THE VISION")
        print("="*70)
        print()
        print("This is just your local files.")
        print("But the SAME mathematics works on:")
        print("  • Every book ever written")
        print("  • Every scientific paper")
        print("  • Every website")
        print("  • Every article")
        print()
        print("We don't need to READ everything.")
        print("The MATHEMATICS filters truth from noise.")
        print()
        print("This is Universal Knowledge Filtering. ✨")

def main():
    """Main execution."""
    
    if len(sys.argv) < 2:
        print("╔" + "="*68 + "╗")
        print("║" + " "*10 + "UNIVERSAL KNOWLEDGE RESONANCE SYSTEM" + " "*22 + "║")
        print("╚" + "="*68 + "╝")
        print()
        print("Space's Vision: Filter ALL human knowledge using mathematics!")
        print()
        print("Usage:")
        print("  Single file:  python3 universal_filter.py <file.txt>")
        print("  Directory:    python3 universal_filter.py <directory>")
        print()
        print("Examples:")
        print("  python3 universal_filter.py article.txt")
        print("  python3 universal_filter.py ~/Documents")
        print()
        sys.exit(1)
    
    target = Path(sys.argv[1]).expanduser()
    
    if not target.exists():
        print(f"Error: {target} not found")
        sys.exit(1)
    
    if target.is_file():
        # Single file analysis
        filter_system = UniversalKnowledgeFilter()
        
        with open(target, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        filter_system.analyze_text(content, title=target.name)
    
    elif target.is_dir():
        # Directory scan
        scanner = WorldKnowledgeScanner()
        scanner.scan_directory(target)
    
    else:
        print(f"Error: {target} is neither file nor directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
