'''
Filename: mpifvm2d-py/src/tools.py
Created Date: Sunday, May 17th 2026, 3:14:51 pm
Author: Lixiang Jiang

All Copyright (c) 2026 Lixiang Jiang. All rights reserved.
'''



def write_tecplot(nodes, elements, val_names, save_pth:str):
    if not save_pth.endswith('.plt'):
        save_pth += '.plt'

    assert nodes.shape[1] == len(val_names), 'Please ensure that the data and name dimensions are consistent.'

    with open(save_pth, 'w') as f:
        f.write('TITLE="Tecplot Unstructured Grid Data"\n')
        f.write('VARIABLES=')
        val_names_str = ''
        for name in val_names:
            val_names_str += name.strip() + ', '
        val_names_str = val_names_str[:-2]
        f.write(val_names_str + '\n')
        f.write('ZONE T="Zone"\n')
        f.write('N={}, E={}, F=FEPOINT, ET=quadrilateral\n'.format(nodes.shape[0], elements.GetTotalNum()))
        f.write('#points:\n')
        for iNode in range(nodes.shape[0]):
            line = ''
            for j in range(nodes.shape[1]):
                line += '{} '.format(nodes[iNode, j])
            f.write(line.strip() + '\n')
        f.write('#elements:\n')
        for iEelem in range(elements.GetTotalNum()):
            if elements.GetNumPart(iEelem) == 4:
                line = ''
                for k in range(elements.GetNumPart(iEelem)):
                    node_idx = elements.GetData(iEelem, k)
                    line += '{} '.format(node_idx + 1)
                f.write(line.strip() + '\n')
            elif elements.GetNumPart(iEelem) == 3:
                line = ''
                for k in range(elements.GetNumPart(iEelem)):
                    node_idx = elements.GetData(iEelem, k)
                    line += '{} '.format(node_idx + 1)
                line += '{} '.format(elements.GetData(iEelem, 0) + 1)
                f.write(line.strip() + '\n')
            else:
                raise ValueError('Unsupported element type with {} nodes.'.format(elements.GetNumPart(iEelem)))