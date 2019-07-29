import os, glob, re
os.chdir(os.path.dirname(os.path.realpath(__file__)))

out=open('README.md','w',encoding='utf-8')
out.write('* files starting with capitalized letters are sample codes for respective topics\n')
out.write('* files starting with lower-case letters are tutorials for packages\n')
out.write('# Table of Contents\n')
out.write('## Topics\n')
switched=False

for fn in sorted(glob.glob('*.ipynb')):
    if fn[0] in 'qwertyuiopasddfghjklzxcvbnm' and not switched:
        switched=True
        out.write('## Packages\n')
    out.write('### [{f}](https://github.com/hoihui/tutorial/blob/master/{f}) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/{f}) [![Viewer](https://camo.githubusercontent.com/3787960e353ddddb30bea9eca931318ff704d6fb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e737667)](https://nbviewer.jupyter.org/github/hoihui/tutorial/blob/master/{f})\n'.format(f=fn))
    with open(fn,'r',encoding='utf-8') as f:
        text = f.read()
    for pre, title in re.findall(r'cell_type[^\w]*markdown[^\w]*metadata": \{[^\}]*[^\w]*source": \[\n\s*"(#{1,2})\s+([^\n]*)"',text):
        s = '  '*len(pre)
        s += '* '+bytes(title,"utf-8", errors='ignore').decode('unicode_escape', errors='ignore').strip()
        s += '\n'
#         print(s)
        out.write(s)
        
out.close()