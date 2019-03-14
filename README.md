# struc2vec
Try to implement the algorithm

## 各代码文件作用

### Graph.py
> 定义了一些关于图的操作
    
    def build_graph(): 建立一个图，是否有向、存在权重可选
    def normalize_weight(): 将权重按比转换成小数
    def get_k_neighbors(self, diameter: int)：diameter: 直径，返回值形式为：
        {
            k: {
                node: 点node的第k跳的邻居节点序列
            }
        }
    def get_sequence_degree(self, k_nbrs: dict): k_nbrs: 上个函数的返回值，返回值形式为：
        {
            k: {
                node: 点node的第k跳邻居节点按入读有小到大排列成的序列
            }
        }
    def get_dtw_distance(s1, s2): s1、s2为两个序列，返回值为s1与s2按dtw算法求出的距离
 
### struc2vec.py
> 定义了具体进行struc2vec的方法

Struc2Vec类中的参数定义如下：

    fk_uv: 记录论文中的u、v的距离的矩阵，为了便于理解可以看作为一个字典
        {
            k: {
                u: {
                    v: fk_uv[k][u][v] 第k层，u、v之间的距离
                }
            }
        }
    wk_uv: 记录第k层u、v的权重的矩阵，为了便于理解，同样可以看作为一个字典
        {
            k: {
                u: {
                    v: wk_uv[k][u][v] 第k层，从u到v的节点权重
                }
            }
        }
    wku: 记录论文中(uk, uk+1)和(uk, uk-1)的权重，是一个三维数组，可以看作是一个字典
        {
            k: {
                u: [w(uk, uk+1), w(uk, uk-1)} 
            }
        }
    wk_average: 记录第k层边权重的平均值，同样可以看作是一个字典
        {
            k: wk_average 第k层边权重的平均值
        }
    tku: u的入射边权重中大于平均权重的个数，可以看作为一个字典
        {
            k: {
                u: thk[k][u] 第k层，u的大于平均权重的入射边的个数
            }
        }
    zku: 第k层u的归一化因子，可以看作为一个字典
        {
            k: {
                u: 
            }
        }
    pk_uv: 第k层节点u走到节点v的概率，可以看作为一个字典
        {
            k: {
                u: {
                    v: 第k层从u到v的概率
                }
            }
        }
    pku: 第k层uk到uk+1或uk到uk-1的概率，可以看作为一个字典
        {
            k: {
                u: [pk(uk, uk+1), pk(uk, uk-1)]
            }
        }
    
    def get_fk_uv(self): 获得fk_uv
    def get_wk_uv(self): 获得wk_uv
    def get_wk_average(self): 获得wk_average
    def get_tku(self): 获得tku
    def get_wku(self): 获得wku
    def get_zku(self): 获得zku
    def get_pk_uv(self): 获得pk_uv
    def get_pku(self): 获得pku



采用先准备好相应数据再进行游走的方法，按顺序执行以下函数：</br>


[1] get_fk_uv
[2] get_wk_uv
[3] 