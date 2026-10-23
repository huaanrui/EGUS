"""Conservative first answer-line extraction, independent of the gold entity."""
import re


def parse_answer_line(text, entities):
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    if not lines:return -1
    line=lines[0].strip(' \t.,:;!?\"\'').lower()
    line=re.sub(r'^answer\s*:\s*','',line)
    narrative=re.fullmatch(r"the pronoun ['\"]?[a-z]+['\"]? refers to (.+)",line)
    if narrative:line=narrative.group(1).strip(' \t.,:;!?\"\'')
    line=re.sub(r'^the\s+','',line)
    return next((i for i,e in enumerate(entities) if line==e.lower()),-1)
