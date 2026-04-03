

def split_chunk(text,chunk_size="\n",overlap_size=0):
    if type(chunk_size) == str:
        chunks = text.split(chunk_size)
        return chunks
    else:
        l = len(text)
        start = 0
        chunks = []
        while start < l:
            end = min(start + chunk_size, l)
            chunks.append(text[start:end])
            if end == l:
                break
            start = end-overlap_size
        return chunks



