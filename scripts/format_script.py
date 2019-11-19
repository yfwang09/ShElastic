import numpy as np

with open('script.py', 'r') as fin:
    findata = fin.read()
with open('submit_python_test2.sbatch', 'r') as fsub:
    fsubdata = fsub.read()

slist = [1, 2, 3, 4]
# blist = [0, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 0.1, 0.3, 0.5, 1, 3, 5]
blist = [1., ]
glist = [1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30]

cmdstr = ''
for shapenum in slist:
    postdata = findata.replace("shapename = 'Shape2'", "shapename = 'Shape%d'"%shapenum)
    for b in blist:
        postdata1 = postdata.replace('mybeta  =', 'mybeta  = %.0e #'%b)
        for g in glist:
            foutdata = postdata1.replace('mygamma =', 'mygamma = %.0e #'%g)
            fileout = 'script_shape%d_b%.0e_g%.0e'%(shapenum, b, g)
            cmdstr=cmdstr+'python3 ' + fileout + '.py > ' + fileout + '.txt &\n'
            with open(fileout + '.py', 'w') as fout:
                fout.write(foutdata)

cmdlines = cmdstr.splitlines()
nfile = len(cmdlines)//24
for i in range(nfile):
    with open('submit_%d.sbatch'%i, 'w') as fsub:
        cmdout = '\n'.join(cmdlines[(i*24):((i+1)*24)])
        fsub.write(fsubdata.replace('# INSERT COMMANDS HERE', cmdout[:-2]))

with open('submit_%d.sbatch'%(nfile), 'w') as fsub:
    cmdout = '\n'.join(cmdlines[nfile*24:])
    fsub.write(fsubdata.replace('# INSERT COMMANDS HERE', cmdout[:-2]))

