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

def init_tags(hotwords: dict[str, str]={}):
    import duckdb
    print('Loading tags database...')
    try:
        db = duckdb.connect()
        query = '''
            SELECT tag FROM (
                SELECT character AS tag, count FROM read_csv('tags/danbooru_character.csv', ignore_errors=true) WHERE count >= 100
                UNION ALL
                SELECT character AS tag, count FROM read_csv('tags/e621_character.csv', ignore_errors=true) WHERE count >= 100
                UNION ALL
                SELECT tags AS tag, count FROM read_csv('tags/danbooru_e621_merged.csv', ignore_errors=true) WHERE count >= 100
            )
            GROUP BY tag
            ORDER BY MAX(count) DESC
        '''
        results = db.execute(query).fetchall()
        tags = [row[0] for row in results]
        # 合并配置区 hotwords 的 key，并去重（保持 hotwords 优先级最高）
        hotword_keys = list(hotwords.keys())
        tags = hotword_keys + [tag for tag in tags if tag not in hotwords]
        print(f'Loaded {len(tags)} tags for completion (including {len(hotword_keys)} hotwords).')
    except Exception as e:
        print(f'Failed to load tags database: {e}')
        # 如果加载失败，至少保留 hotwords 的 key
        tags = list(hotwords.keys())
        
    return TagCompleter(tags)