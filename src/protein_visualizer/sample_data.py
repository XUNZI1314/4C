PDB_TEXT = '''
ATOM      1  N   ALA A   1      11.104  13.207   7.115  1.00 20.00           N  
ATOM      2  CA  ALA A   1      12.560  13.404   7.275  1.00 20.00           C  
ATOM      3  C   ALA A   1      13.081  12.990   8.657  1.00 20.00           C  
ATOM      4  O   ALA A   1      12.424  12.261   9.405  1.00 20.00           O  
ATOM      5  CB  ALA A   1      13.050  14.853   6.997  1.00 20.00           C  
ATOM      6  N   TYR A   2      14.284  13.461   9.006  1.00 18.00           N  
ATOM      7  CA  TYR A   2      14.893  13.140  10.293  1.00 18.00           C  
ATOM      8  C   TYR A   2      15.287  11.667  10.426  1.00 18.00           C  
ATOM      9  O   TYR A   2      15.329  10.929   9.437  1.00 18.00           O  
ATOM     10  CB  TYR A   2      16.129  14.031  10.515  1.00 18.00           C  
ATOM     11  CG  TYR A   2      15.889  15.501  10.330  1.00 18.00           C  
ATOM     12  CD1 TYR A   2      15.005  16.169  11.183  1.00 18.00           C  
ATOM     13  CD2 TYR A   2      16.548  16.220   9.326  1.00 18.00           C  
ATOM     14  CE1 TYR A   2      14.792  17.521  11.010  1.00 18.00           C  
ATOM     15  CE2 TYR A   2      16.341  17.569   9.146  1.00 18.00           C  
ATOM     16  CZ  TYR A   2      15.463  18.207  10.006  1.00 18.00           C  
ATOM     17  OH  TYR A   2      15.257  19.548   9.827  1.00 18.00           O  
ATOM     18  N   GLU A   3      15.585  11.251  11.662  1.00 16.00           N  
ATOM     19  CA  GLU A   3      15.923   9.859  11.973  1.00 16.00           C  
ATOM     20  C   GLU A   3      17.326   9.470  11.530  1.00 16.00           C  
ATOM     21  O   GLU A   3      18.235  10.274  11.700  1.00 16.00           O  
ATOM     22  CB  GLU A   3      15.702   9.575  13.470  1.00 16.00           C  
ATOM     23  CG  GLU A   3      14.292   9.924  13.934  1.00 16.00           C  
ATOM     24  CD  GLU A   3      14.051   9.611  15.401  1.00 16.00           C  
ATOM     25  OE1 GLU A   3      14.967   9.805  16.221  1.00 16.00           O  
ATOM     26  OE2 GLU A   3      12.941   9.167  15.737  1.00 16.00           O  
ATOM     27  N   LYS A   4      17.493   8.242  10.972  1.00 15.00           N  
ATOM     28  CA  LYS A   4      18.782   7.732  10.487  1.00 15.00           C  
ATOM     29  C   LYS A   4      19.026   6.305  10.970  1.00 15.00           C  
ATOM     30  O   LYS A   4      18.118   5.479  10.987  1.00 15.00           O  
ATOM     31  CB  LYS A   4      18.846   7.806   8.954  1.00 15.00           C  
ATOM     32  CG  LYS A   4      20.226   7.495   8.373  1.00 15.00           C  
ATOM     33  CD  LYS A   4      20.255   7.622   6.858  1.00 15.00           C  
ATOM     34  CE  LYS A   4      21.664   7.359   6.343  1.00 15.00           C  
ATOM     35  NZ  LYS A   4      21.692   7.523   4.869  1.00 15.00           N  
ATOM     36  N   VAL A   5      20.265   6.026  11.356  1.00 17.00           N  
ATOM     37  CA  VAL A   5      20.671   4.705  11.865  1.00 17.00           C  
ATOM     38  C   VAL A   5      21.337   3.880  10.777  1.00 17.00           C  
ATOM     39  O   VAL A   5      22.239   4.344  10.081  1.00 17.00           O  
ATOM     40  CB  VAL A   5      21.626   4.815  13.080  1.00 17.00           C  
ATOM     41  CG1 VAL A   5      22.969   5.463  12.689  1.00 17.00           C  
ATOM     42  CG2 VAL A   5      21.893   3.443  13.683  1.00 17.00           C  
TER
END
'''

MMPBSA_TEXT = '''
Resid  Chain Residue    DELTA TOTAL
1      A     ALA       -12.5
2      A     TYR        -7.2
3      A     GLU         1.8
4      A     LYS         6.4
5      A     VAL        11.1
'''


POCKET_TEXT = '''
pocket_id,chain,resid,resname,volume,score
Pocket-1,A,1,ALA,268.5,0.91
Pocket-1,A,2,TYR,268.5,0.91
Pocket-1,A,3,GLU,268.5,0.91
Pocket-2,A,4,LYS,182.2,0.73
Pocket-2,A,5,VAL,182.2,0.73
'''


ANNOTATION_TEXT = '''
chain,resid,resname,annotation,region_type
A,1,ALA,入口邻近残基,binding-rim
A,2,TYR,候选热点残基,hotspot
A,3,GLU,口袋边缘残基,pocket-edge
A,4,LYS,界面波动残基,interface
A,5,VAL,构象敏感残基,flexible
'''


PDB_TEXT_ALT = '''
ATOM      1  N   ALA A   1      10.904  13.007   6.915  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.360  13.204   7.075  1.00 20.00           C
ATOM      3  C   ALA A   1      12.981  12.790   8.457  1.00 20.00           C
ATOM      4  O   ALA A   1      12.324  12.061   9.205  1.00 20.00           O
ATOM      5  CB  ALA A   1      12.950  14.653   6.797  1.00 20.00           C
ATOM      6  N   TYR A   2      14.084  13.261   8.806  1.00 18.00           N
ATOM      7  CA  TYR A   2      14.693  12.940  10.093  1.00 18.00           C
ATOM      8  C   TYR A   2      15.087  11.467  10.226  1.00 18.00           C
ATOM      9  O   TYR A   2      15.129  10.729   9.237  1.00 18.00           O
ATOM     10  CB  TYR A   2      15.929  13.831  10.315  1.00 18.00           C
ATOM     11  CG  TYR A   2      15.689  15.301  10.130  1.00 18.00           C
ATOM     12  CD1 TYR A   2      14.805  15.969  10.983  1.00 18.00           C
ATOM     13  CD2 TYR A   2      16.348  16.020   9.126  1.00 18.00           C
ATOM     14  CE1 TYR A   2      14.592  17.321  10.810  1.00 18.00           C
ATOM     15  CE2 TYR A   2      16.141  17.369   8.946  1.00 18.00           C
ATOM     16  CZ  TYR A   2      15.263  18.007   9.806  1.00 18.00           C
ATOM     17  OH  TYR A   2      15.057  19.348   9.627  1.00 18.00           O
ATOM     18  N   GLU A   3      15.385  11.051  11.462  1.00 16.00           N
ATOM     19  CA  GLU A   3      15.723   9.659  11.773  1.00 16.00           C
ATOM     20  C   GLU A   3      17.126   9.270  11.330  1.00 16.00           C
ATOM     21  O   GLU A   3      18.035  10.074  11.500  1.00 16.00           O
ATOM     22  CB  GLU A   3      15.502   9.375  13.270  1.00 16.00           C
ATOM     23  CG  GLU A   3      14.092   9.724  13.734  1.00 16.00           C
ATOM     24  CD  GLU A   3      13.851   9.411  15.201  1.00 16.00           C
ATOM     25  OE1 GLU A   3      14.767   9.605  16.021  1.00 16.00           O
ATOM     26  OE2 GLU A   3      12.741   8.967  15.537  1.00 16.00           O
ATOM     27  N   LYS A   4      17.293   8.042  10.772  1.00 15.00           N
ATOM     28  CA  LYS A   4      18.582   7.532  10.287  1.00 15.00           C
ATOM     29  C   LYS A   4      18.826   6.105  10.770  1.00 15.00           C
ATOM     30  O   LYS A   4      17.918   5.279  10.787  1.00 15.00           O
ATOM     31  CB  LYS A   4      18.646   7.606   8.754  1.00 15.00           C
ATOM     32  CG  LYS A   4      20.026   7.295   8.173  1.00 15.00           C
ATOM     33  CD  LYS A   4      20.055   7.422   6.658  1.00 15.00           C
ATOM     34  CE  LYS A   4      21.464   7.159   6.143  1.00 15.00           C
ATOM     35  NZ  LYS A   4      21.492   7.323   4.669  1.00 15.00           N
ATOM     36  N   VAL A   5      20.065   5.826  11.156  1.00 17.00           N
ATOM     37  CA  VAL A   5      20.471   4.505  11.665  1.00 17.00           C
ATOM     38  C   VAL A   5      21.137   3.680  10.577  1.00 17.00           C
ATOM     39  O   VAL A   5      22.039   4.144   9.881  1.00 17.00           O
ATOM     40  CB  VAL A   5      21.426   4.615  12.880  1.00 17.00           C
ATOM     41  CG1 VAL A   5      22.769   5.263  12.489  1.00 17.00           C
ATOM     42  CG2 VAL A   5      21.693   3.243  13.483  1.00 17.00           C
TER
END
'''


MMPBSA_TEXT_ALT = '''
Resid  Chain Residue    DELTA TOTAL
1      A     ALA       -10.1
2      A     TYR        -8.9
3      A     GLU        -2.4
4      A     LYS         4.8
5      A     VAL         9.7
'''
