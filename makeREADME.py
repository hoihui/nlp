import os, glob, re
os.chdir(os.path.dirname(os.path.realpath(__file__)))

out=open('README.md','w')
out.write('* files starting with capitalized letters are sample codes for respective topics\n')
out.write('* files starting with lower-case letters are tutorials for packages\n')
out.write('# Table of Contents\n')
out.write('# Topics\n')
switched=False

for fn in sorted(glob.glob('*.ipynb')):
    if fn[0] in 'qwertyuiopasddfghjklzxcvbnm' and not switched:
        switched=True
        out.write('# Packages\n')
    out.write('### <a href="https://github.com/hoihui/pkgs/blob/master/{f}">{f}</a>\n'.format(f=fn))
    with open(fn) as f:
        text = f.read()
    for pre, title in re.findall(r'cell_type[^\w]*markdown[^\w]*metadata": \{[^\}]*[^\w]*source": \[\n\s*"(#{1,2})\s+([^\n]*)"',text):
        out.write('  '*len(pre)+'* '+bytes(title,"utf-8").decode('unicode_escape')+'\n')
        
out.close()