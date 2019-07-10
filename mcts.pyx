# %%cython -+ -a
cimport cython
from copy import deepcopy
import time, random
from libc.math cimport sqrt, log
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement
from numpy.math cimport INFINITY as inf
import multiprocessing as mp
import numpy as np

cdef class Game:
    cpdef list GetMoves(self): return []
    cpdef void Move(self,int move):pass
    cpdef void RollOut(self):pass
    cpdef str key(self): return "";
    cpdef double GetResult(self,int viewpoint): return 0.5;

cdef class MCTS():
    def __init__(self, Game game):
        self.nodes = {}
        self.game = game

    def search(self, timemax=None, itermax=None, explore=1, verbose=0):
        rootkey=self.game.key()
        cpus = mp.cpu_count()
        pool = mp.Pool(cpus)
        if 'Reachable' in dir(self.game): #pruning by (game provided) finding which node will be reachable
            self.nodes = {k:v for k,v in self.nodes.items() if k==rootkey or self.game.Reachable(k)}

        nextmoves =  self.rootnode['untriedMoves']+list(self.rootnode['childNodes'])
        if len(nextmoves)>1:  #unimportant -- determine when to stop
            if itermax or timemax:
                start=time.time()
                i = 0
                OneBatch=100000000
                if timemax: simulate_multiple((self.game,self.nodes,0)); OneBatch=int(timemax/(time.time()-start))
                if itermax: OneBatch = min(OneBatch,max(1,itermax//3))
                while (timemax is None or time.time()<start+timemax) and\
                      (itermax is None or i<itermax):
                    i = i+OneBatch
                    li=list(range(OneBatch//cpus*100,OneBatch//cpus*100+cpus))
                    vs,nodes = zip(*pool.map(simulate_multiple,[(self.game,self.nodes,e) for e in li]))
                    v=max(vs)                    
                    for d in nodes:     # subtract the input wins and visits from the output
                        for key, n in d.items():
                            if key in self.nodes:
                                selfnode = self.nodes[key]
                                if np.isfinite(selfnode['wins']):
                                    n['wins']-=selfnode['wins']
                                n['visits']-=selfnode['visits']
                    for d in nodes:     # add the generated wins and visits to the record
                        for key, n in d.items():
                            if key in self.nodes:
                                selfnode = self.nodes[key]
                                selfnode['wins'] += n['wins']
                                selfnode['visits'] += n['visits']
                                if set(n['childNodes'])-set(selfnode['childNodes']):
                                    selfnode['childNodes'].update(n['childNodes'])
                                    selfnode['untriedMoves'] = [mv for mv in selfnode['untriedMoves'] if mv not in selfnode['childNodes']]
                            else:
                                self.nodes[key]=deepcopy(n)
                    if abs(v)==inf: break
            else:
                start=time.time()
                while True:
                    vs,nodes = zip(*pool.map(simulate_multiple,
                                             [(self.game,self.nodes,e) for e in range(1000*100,1000*100+cpus)]))
                    v=max(vs)
                    wins = {}      #(for speed), just choose a node and update the nextlevel childNodes counts only from other returned nodes
                    visits = {}
                    for mv,key in self.nodes[rootkey]['childNodes'].items():
                        if np.isfinite(self.nodes[key]['wins']):
                            wins[mv] = self.nodes[key]['wins']
                        visits[mv] = self.nodes[key]['visits']                    
                    self.nodes=nodes[0]
                    for d in nodes[1:]:
                        for mv,key in d[rootkey]['childNodes'].items():
                            if key in self.nodes:
                                self.nodes[key]['wins']+=d[key]['wins']-wins.get(mv,0)
                                self.nodes[key]['visits']+=d[key]['visits']-visits.get(mv,0)
                    
                    print(time.time()-start,sum(self.nodes[k]['visits'] for k in self.rootnode['childNodes'].values()),
                          self.ChildrenToString(self.rootnode).replace('\n',' '),
                          end='\r')
                    if abs(v)==inf: break
                    
        if len(nextmoves)>1:
            moveToChild = self.UCTSelectMove(self.rootnode,self.nodes,explore=0)
        else:
            moveToChild = nextmoves[0]
        if verbose: 
            print(self.ChildrenToString(self.rootnode))
            print('#simulations:',sum(self.nodes[k]['visits'] for k in self.rootnode['childNodes'].values()))
        pool.close()
        return moveToChild
    def ChildrenToString(self,node):
        return "\n".join(f'{str(mv):6s}-> [W/V: {n["wins"]:6g}/{n["visits"]:6.0f} | UnXplrd: {len(n["untriedMoves"])}]'
                            for mv,k in sorted(node['childNodes'].items(),key=lambda e: self.nodes[e[1]]['wins']/self.nodes[e[1]]['visits'])
                            for n in [self.nodes[k]]
                           )
    @property
    def rootnode(self):
        key = self.game.key()
        if key not in self.nodes:
            self.nodes[key] = {'viewpoint':3-self.game.playerToMove,
                               'wins':0.,
                               'visits':0,
                               'childNodes':{},
                               'untriedMoves':self.game.GetMoves()}
        return self.nodes[key]
        
    cdef int UCTSelectMove(self,n,nodes,explore=1):
        cdef double bestval
        cdef int bestmv = -191247
        if nodes is None: nodes=self.nodes
        for mv,k in n['childNodes'].items():
            nn = nodes[k]
            curval = nn['wins']/nn['visits'] + explore*sqrt(log(n['visits'])/nn['visits'])
            if bestmv == -191247 or curval>bestval:
                bestmv = mv
                bestval = curval
        return bestmv
    
def simulate_multiple(p):
    game,nodes,i=p
    rootkey = game.key()
    node=nodes[rootkey]
    np.random.seed(i)
    random.seed(i)
    OneBatch,i=divmod(i,100)
    greatestval = -inf
    for _ in range(OneBatch+1):
        v = simulate(deepcopy(game),node,nodes)
        if v>greatestval:greatestval=v
    return greatestval,nodes
    
cdef double simulate(game,node,nodes,explore=1):
    cdef double v
    cdef double bestval, curval
    cdef int move = -191247
    if node['untriedMoves'] == [] and node['childNodes'] != {}: #fully expanded, non-terminal
        
        for mv,k in node['childNodes'].items():
            nn = nodes[k]
            curval = nn['wins']/nn['visits'] + explore*sqrt(log(node['visits'])/nn['visits'])
            if move == -191247 or curval>bestval:
                move = mv
                bestval = curval
    
        nnode = nodes[node['childNodes'][move]]
        if nnode['wins']==inf or nnode['wins']==-inf: #if the best is -inf or inf already, can backprop
            node['wins']+= -nnode['wins']; node['visits']+=1
            return -nnode['wins']
        game.Move(move)
        v = simulate(game,nnode,nodes)
        if v==-inf: v=0 # if v==-inf, next mover will not choose this branch as it leads to loss.
                             # Note: other branch may have finite v as UCT  saw this branch as finite v before simulation
        node['wins']+=1-v; node['visits']+=1;
        return 1-v
    elif node['untriedMoves'] != []: #not fully expanded: expand and then rollout
        move = random.choice(node['untriedMoves']) 
        game.Move(move)
        k = game.key()
        if k not in nodes:
            nodes[k] = { 'viewpoint':3-game.playerToMove,
                         'wins':0.,
                         'visits':0.,
                         'childNodes':{},
                         'untriedMoves':game.GetMoves()}
        nnode = nodes[k]
        node['untriedMoves'].remove(move)
        node['childNodes'][move]=k
        game.RollOut()
        v = game.GetResult(nnode['viewpoint']) #not setting inf or -inf because there are indeterminism (unexplored moves)
        nnode['wins']+=v; nnode['visits']+=1
        node['wins']+=1-v; node['visits']+=1
        return 1-v
    elif node['childNodes'] == {}: #terminal
        v = game.GetResult(node['viewpoint'])
        if v==1:    v=inf
        elif v==0:  v=-inf
        node['wins']+=v; node['visits']+=1
        return v
