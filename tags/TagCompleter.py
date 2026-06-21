from prompt_toolkit.completion import Completer, Completion

class TagCompleter(Completer):
    def __init__(self, words):
        self.words = words

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        parts = text.split(',')
        if not parts:
            return
        
        current_part = parts[-1]
        current_word = current_part.lstrip()
        start_position = -len(current_word)
        
        if not current_word:
            return
            
        count = 0
        prefix_matches = []
        
        # 1. 前缀匹配
        for word in self.words:
            if word.lower().startswith(current_word.lower()):
                prefix_matches.append(word)
                yield Completion(word, start_position=start_position)
                count += 1
                if count >= 20:
                    break
                    
        # 2. 包含匹配
        if count < 20:
            for word in self.words:
                if word not in prefix_matches and current_word.lower() in word.lower():
                    yield Completion(word, start_position=start_position)
                    count += 1
                    if count >= 20:
                        break