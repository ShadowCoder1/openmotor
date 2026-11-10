"""Minimal reader for the .xlsx files of the Zaidi deposit (Excel strict-OOXML, which openpyxl cannot open) and .xls files. Reads data only."""
import zipfile, re
import xml.etree.ElementTree as ET
def _t(e): return e.tag.rsplit('}',1)[-1]
def colnum(ref):
    s=re.match(r'[A-Z]+',ref).group(0); n=0
    for ch in s: n=n*26+ord(ch)-64
    return n-1
def read_xlsx(path):
    """returns {sheetname: list of rows (lists)} for strict or transitional xlsx"""
    z=zipfile.ZipFile(path)
    ss=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')):
            ss.append(''.join(x.text or '' for x in si.iter() if _t(x)=='t'))
    wb=ET.fromstring(z.read('xl/workbook.xml'))
    rels={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    out={}
    for sh in wb.iter():
        if _t(sh)!='sheet': continue
        rid=[v for k,v in sh.attrib.items() if k.endswith('}id')][0]
        tgt=rels[rid].lstrip('/'); tgt=tgt if tgt.startswith('xl/') else 'xl/'+tgt
        rows=[]
        with z.open(tgt) as f:
            for ev,e in ET.iterparse(f):
                if _t(e)=='row':
                    row=[]
                    for c in e:
                        if _t(c)!='c': continue
                        i=colnum(c.get('r')); 
                        while len(row)<=i: row.append(None)
                        v=None
                        for ch in c:
                            if _t(ch)=='v': v=ch.text
                            elif _t(ch)=='is': v=''.join(x.text or '' for x in ch.iter() if _t(x)=='t')
                        if v is not None:
                            t=c.get('t')
                            if t=='s': v=ss[int(v)]
                            elif t in('str','inlineStr','e'): pass
                            elif t=='b': v=bool(int(v))
                            else: v=float(v)
                        row[i]=v
                    rows.append(row); e.clear()
        out[sh.get('name')]=rows
    return out
def read_xls(path):
    import xlrd
    wb=xlrd.open_workbook(path)
    return {s.name:[s.row_values(i) for i in range(s.nrows)] for s in wb.sheets()}
def read_any(path):
    return read_xlsx(path) if str(path).endswith('.xlsx') else read_xls(path)
