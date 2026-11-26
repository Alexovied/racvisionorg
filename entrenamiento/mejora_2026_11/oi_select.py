import csv,collections,json,random
H='/m/04hgtk'
HATS={'/m/02dl1y','/m/025rp__','/m/02wbtzl','/m/02fq_6','/m/02jfl0','/m/04tn4x'}
HELM={'/m/03p3bw','/m/07qxg_','/m/0zvk5'}
sel=[]
random.seed(0)
for split,f in [('validation','validation-annotations-bbox.csv'),('test','test-annotations-bbox.csv')]:
  d=collections.defaultdict(list)
  for r in csv.DictReader(open(f)): d[r['ImageID']].append(r)
  plain=[]
  for i,rs in d.items():
    labs={r['LabelName'] for r in rs}
    if H not in labs or labs&HELM: continue
    heads=[r for r in rs if r['LabelName']==H]
    if any(r['IsGroupOf']=='1' or r['IsDepiction']=='1' for r in heads): continue
    if len(heads)>8: continue
    big=max((float(r['XMax'])-float(r['XMin'])) for r in heads)
    if big<0.04: continue
    hats=[r for r in rs if r['LabelName'] in HATS and r['IsGroupOf']=='0']
    item={'id':i,'split':split,'heads':[[float(r[k]) for k in ('XMin','YMin','XMax','YMax')] for r in heads],
          'hats':[[float(r[k]) for k in ('XMin','YMin','XMax','YMax')] for r in hats]}
    if hats: sel.append(item)
    else: plain.append(item)
  random.shuffle(plain); sel+=plain[:900]
json.dump(sel,open('seleccion.json','w'))
print(len(sel), sum(1 for s in sel if s['hats']))
