"""Collect citation metadata from primary publication pages; no experiment access."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import urllib.request

OUT = Path(__file__).resolve().parent

# key, metadata page, topic, optional verified proceedings year/title/URL.
CATALOG = [
    ('bbq2022', 'https://aclanthology.org/2022.findings-acl.165/', 'evaluation', {}),
    ('rethinking2025', 'https://aclanthology.org/2025.findings-acl.1361/', 'evaluation', {}),
    ('oneexample2026', 'https://arxiv.org/abs/2609.14860', 'evaluation', {}),
    ('synthbias2025', 'https://proceedings.mlr.press/v267/wang25cp.html', 'evaluation', {}),
    ('ceb2025', 'https://arxiv.org/abs/2407.02408', 'evaluation', {'year':'2025','booktitle':'International Conference on Learning Representations', 'venue_source':'https://github.com/SongW-SW/CEB'}),
    ('openbbq2024', 'https://arxiv.org/abs/2412.06134', 'evaluation', {}),
    ('repe2023', 'https://arxiv.org/abs/2310.01405', 'steering', {}),
    ('caa2024', 'https://aclanthology.org/2024.acl-long.828/', 'steering', {}),
    ('refusal2024', 'https://arxiv.org/abs/2406.11717', 'steering', {'year':'2024','booktitle':'Advances in Neural Information Processing Systems', 'url':'https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html'}),
    ('cast2025', 'https://arxiv.org/abs/2409.05907', 'steering', {'year':'2025','booktitle':'International Conference on Learning Representations','venue_source':'https://github.com/IBM/activation-steering'}),
    ('fairsteer2025', 'https://arxiv.org/abs/2504.14492', 'steering', {}),
    ('knowbias2026', 'https://proceedings.mlr.press/v306/pan26j.html', 'steering', {}),
    ('crispr2024', 'https://aclanthology.org/2024.acl-long.490/', 'steering', {}),
    ('bba2026', 'https://arxiv.org/abs/2602.04398', 'steering', {'year':'2026','booktitle':'International Conference on Learning Representations','url':'https://proceedings.iclr.cc/paper_files/paper/2026/hash/871ed095b734818cfba48db6aeb25a62-Abstract-Conference.html'}),
    ('axbench2025', 'https://proceedings.mlr.press/v267/wu25a.html', 'controls', {}),
    ('saefeatures2025', 'https://aclanthology.org/2025.emnlp-main.519/', 'controls', {}),
    ('selectiveqa2020', 'https://aclanthology.org/2020.acl-main.503/', 'abstention', {}),
    ('sufficient2025', 'https://arxiv.org/abs/2411.06037', 'abstention', {'year':'2025','booktitle':'International Conference on Learning Representations','url':'https://proceedings.iclr.cc/paper_files/paper/2025/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html'}),
    ('internal2023', 'https://aclanthology.org/2023.findings-emnlp.68/', 'probing', {}),
    ('know2022', 'https://arxiv.org/abs/2207.05221', 'probing', {}),
    ('probecontrols2019', 'https://aclanthology.org/D19-1275/', 'probing', {}),
    ('yopo2026', 'https://arxiv.org/abs/2608.14465', 'abstention', {}),
    ('adaptive2026', 'https://arxiv.org/abs/2608.25379', 'steering', {}),
    ('squad2018', 'https://aclanthology.org/P18-2124/', 'abstention', {}),
]


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta = {}

    def handle_starttag(self, tag, attrs):
        item = dict(attrs)
        name = item.get('name', '')
        if tag == 'meta' and name.startswith('citation_') and name != 'citation_abstract':
            self.meta.setdefault(name, []).append(item.get('content', ''))


def fetch(item):
    key, url, topic, overrides = item
    with urllib.request.urlopen(url, timeout=45) as response:
        raw = response.read()
        resolved = response.url
    parser = Metadata()
    parser.feed(raw.decode('utf-8'))
    meta = parser.meta
    def first(*names):
        return next((meta[n][0] for n in names if n in meta), '')
    title = first('citation_title')
    authors = meta.get('citation_author', [])
    date = first('citation_publication_date', 'citation_date', 'citation_online_date')
    year = re.search(r'\d{4}', date)
    assert title and authors and year, (key, meta)
    record = {'key':key,'topic':topic,'title':title,'authors':authors,
              'year':year.group(), 'url':url, 'metadata_url':url,
              'resolved_url':resolved, 'page_sha256':hashlib.sha256(raw).hexdigest(),
              'accessed_utc':datetime.now(timezone.utc).isoformat(), 'metadata':meta,
              'booktitle':first('citation_conference_title','citation_inbook_title'),
              'doi':first('citation_doi'), 'arxiv_id':first('citation_arxiv_id'),
              'pages':'--'.join(filter(None,[first('citation_firstpage'),first('citation_lastpage')]))}
    record.update(overrides)
    record['citation_type'] = 'inproceedings' if record['booktitle'] else 'misc'
    return record


def escape(s):
    return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def bibtex(r):
    # ACL HTML metadata may append native-script aliases in parentheses;
    # preserve raw provenance in sources.json but use the official Latin names
    # in the manuscript bibliography (matching the page's BibTeX export).
    author_aliases = {'Runzhe Zhan (詹润哲)': 'Runzhe Zhan',
                      'Derek F. Wong (黄辉)': 'Derek F. Wong'}
    authors = [author_aliases.get(a, a) for a in r['authors']]
    fields = {'title':'{'+escape(r['title'])+'}', 'author':' and '.join(map(escape,authors)),
              'year':r['year'], 'url':r['url']}
    if r['booktitle']: fields['booktitle'] = escape(r['booktitle'])
    if r['pages']: fields['pages'] = r['pages']
    if r['doi']: fields['doi'] = r['doi']
    if r['arxiv_id']:
        fields.update(eprint=r['arxiv_id'],archivePrefix='arXiv')
    if r['citation_type']=='misc': fields['note']='arXiv version cited'
    return '@'+r['citation_type']+'{'+r['key']+',\n'+''.join('  '+k+' = {'+v+'},\n' for k,v in fields.items())+'}\n'


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, CATALOG))
    assert len({r['key'] for r in records}) == len(records)
    (OUT/'sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'references.bib').write_text('% Primary-source metadata checked 2026-10-17. See sources.json for provenance.\n\n'+'\n'.join(map(bibtex,records)),encoding='utf-8')
    print(json.dumps({'references':len(records),'titles':[{k:r[k] for k in ('key','title','year','booktitle')} for r in records]},ensure_ascii=False))
