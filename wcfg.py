
import sys
import re
import random
import functools
import itertools
import math
import time
import graphviz
import numpy as np
from tabulate import tabulate
from collections import defaultdict, Counter

def product(lst):
    return functools.reduce(lambda x,y: x*y, lst, 1)

def fixed_point(f, start, equal, callback=None):
    current = start
    while True:
        if callback is not None:
            callback(current)
        previous = current
        current = f(previous)
        if equal(previous, current):
            break
    return current

# Depth-first search, with callbacks for when a node is discovered, when a node is 
# finished, when a node is discovered as a root of the forest, and when we descend from 
# a parent to its child in the forest.
def dfscb(vertices, edges, cb_discovered = lambda v,t: None, 
                           cb_finished = lambda v,t: None, 
                           cb_root = lambda v: None, 
                           cb_descend = lambda u,v: None):

    vertices = list(vertices)  # in case we've been given an iterator
    assert all((x in vertices and y in vertices) for (x,y) in edges)

    discovered = set([])
    time = 0

    def dfs_visit(u,t):
        t = t+1
        cb_discovered(u,t)
        discovered.add(u)
        for v in vertices:
            if (u,v) in edges:
                if v not in discovered:
                    cb_descend(u,v)
                    t = dfs_visit(v,t)
        t = t+1
        cb_finished(u,t)
        return t

    for u in vertices:
        if u not in discovered:
            cb_root(u)
            time = dfs_visit(u,time)

# Algorithm from Cormen et al, Third Edition, p.617
def strongly_connected_components(vertices, edges):

    # First DFS: collect the vertices into a list in the order in which they are finished
    finished_list = []
    dfscb(vertices, edges, cb_finished = lambda v,t: finished_list.append(v))

    # Second DFS: with finished_list reversed and the edges reversed, the resulting forest will correspond to SCCs
    components = []
    def add_to_component(u,v):
        [comp] = [c for c in components if u in c]
        comp.append(v)
    dfscb(reversed(finished_list), [(y,x) for (x,y) in edges], 
                   cb_root = lambda v: components.append([v]), 
                   cb_descend = add_to_component)

    return components

# vertices is a list of vertices; edge_weights is a dictionary mapping pairs of vertices to weights
def shortest_paths(vertices, edge_weights):

    dist = defaultdict(lambda: 0)

    for (v,w) in edge_weights.keys():
        dist[(v,w)] = edge_weights[(v,w)]

    for v in vertices:
        dist[(v,v)] = 1

    for v_k in vertices:
        for v_i in vertices:
            for v_j in vertices:
                dist[(v_i,v_j)] = max(dist[(v_i,v_j)], dist[(v_i,v_k)] * dist[(v_k,v_j)])

    return dist

############################################################################################

class DivergenceException(Exception):
    pass

############################################################################################

class WeightedCFG:

    ###################################################################################

    # nontermrules is a sequence of (lhs,rhs,w) triples, where lhs is a nonterminal, rhs is 
    #       a (non-empty) sequence of nonterminals, and w is a weight.
    # termrules is a sequence of (lhs,t,w) triples, where lhs is a nonterminal, t is a 
    #       terminal, and w is a weight.
    def __init__(self, nontermrules, termrules, start_symbol=None):

        if start_symbol is None:
            (self._start_symbol, _, _) = nontermrules[0]
        else:
            self._start_symbol = start_symbol

        # We collect the rules into dictionaries mapping (lhs,rhs) pairs to weights, summing weights where necessary. 
        # Then convert back to lists of tuples at the end, with no duplicate rules remaining.
        nontermrules_dict = defaultdict(lambda: 0)
        termrules_dict = defaultdict(lambda: 0)
        for (lhs,rhs,w) in nontermrules:
            nontermrules_dict[(lhs,tuple(rhs))] += w
        for (lhs,rhs,w) in termrules:
            termrules_dict[(lhs,rhs)] += w
        self._nontermrules = [(lhs,list(rhs),w) for ((lhs,rhs),w) in nontermrules_dict.items()]
        self._termrules = [(lhs,rhs,w) for ((lhs,rhs),w) in termrules_dict.items()]

        for (lhs,rhs,w) in self._nontermrules:
            assert self._start_symbol not in rhs, "Start symbol %s appears on right-hand side of rule %s" % (self._start_symbol, (lhs,rhs,w))

        all_zeros = {nt : 0 for nt in self.all_nonterminals()}
        threshold = 1e-9
        close_enough = lambda d1, d2: all([abs(d1[nt] - d2[nt]) < threshold for nt in self.all_nonterminals()])
        def check(d):
            if math.inf in d.values():
                raise DivergenceException("Infinite value")

        # print("Before calculating inside:", time.perf_counter())
        self._id = fixed_point(lambda d: self._update_inside_dict(d), all_zeros, close_enough, check)
        # print("Finished inside:          ", time.perf_counter())
        self._od = fixed_point(lambda d: self._update_outside_dict(self._id, d), all_zeros, close_enough, check)
        # print("Finished outside v1:      ", time.perf_counter())
        self._od2 = self.solve_outside_system(lambda nt: self._id[nt])
        # print("Finished outside v2:      ", time.perf_counter())

    # previous is a dictionary mapping nonterminals to weights; return value is another dictionary like that
    def _update_inside_dict(self, previous):
        result = {nt : 0 for nt in self.all_nonterminals()}
        for (lhs,rhs,wt) in self._nontermrules:
            result[lhs] += wt * product([previous[nt] for nt in rhs])
        for (lhs,rhs,wt) in self._termrules:
            result[lhs] += wt
        return result

    # previous is a dictionary mapping nonterminals to weights; return value is another dictionary like that
    def _update_outside_dict(self, inside_dict, previous):
        result = {nt : 0 for nt in self.all_nonterminals()}
        result[self._start_symbol] += 1
        for (lhs,rhs,wt) in self._nontermrules:
            for (i,nt) in enumerate(rhs):
                result[nt] += previous[lhs] * wt * product([inside_dict[x] for x in rhs[:i]]) * product([inside_dict[x] for x in rhs[(i+1):]])
        return result

    def solve_outside_system(self, inside):

        nts = self.all_nonterminals()

        coeffs = defaultdict(lambda: 0)         # coeffs[(x,y)] will be the coefficient of outside(y) in the equation for outside(x)
        constants = defaultdict(lambda: 0)      # constants[x] will be the constant in the equation for outside(x)

        for nt in nts:
            coeffs[(nt,nt)] = -1
        constants[self._start_symbol] = 1

        for (lhs,rhs,wt) in self._nontermrules:
            for (i,x) in enumerate(rhs):
                coeffs[(x,lhs)] += wt * product([inside(y) for y in (rhs[:i] + rhs[(i+1):])])

        a = np.array([[coeffs[(x,y)] for y in nts] for x in nts])
        b = - np.array([constants[x] for x in nts])
        soln = np.linalg.solve(a, b)
        d = {nt : x for (nt,x) in zip(nts,soln)}

        return d

    ###################################################################################

    # Each element is pairs is of the form (g,nt), where g is a grammar and nt is the 
    # nonterminal to use as the root/start symbol of g's part of the concatenation grammar. 
    # Each grammar must be top-down normalized; this ensures that the chosen root/start symbols 
    # all have inside values of 1, and the resulting grammar is therefore also top-down normalized.
    def concat_grammars(pairs):

        assert pairs != []
        assert all(g.is_top_down_normalized() for (g,nt) in pairs)

        startsymbol = ("ROOT", -1)
        top_rule = (startsymbol, [(nt,i) for (i,(g,nt)) in enumerate(pairs)], 1)
        nontermrules = [top_rule]
        termrules = []

        def tag_nontermrule(lhs,rhs,wt,i):
            return ((lhs,i), [(nt,i) for nt in rhs], wt)

        def tag_termrule(lhs,rhs,wt,i):
            return ((lhs,i), rhs, wt)

        for (i,(g,nt)) in enumerate(pairs):
            nontermrules += [tag_nontermrule(*r,i) for r in g._nontermrules]
            termrules += [tag_termrule(*r,i) for r in g._termrules]

        g = WeightedCFG(nontermrules, termrules, startsymbol)
        g.remove_useless_rules()
        return g

    ###################################################################################

    def all_nonterminals(self):
        from_lhss = [lhs for (lhs,_,_) in self._nontermrules + self._termrules]
        from_rhss = [nt for (_,rhs,_) in self._nontermrules for nt in rhs]
        return sorted(set(from_lhss+from_rhss))

    def nonterm_expansions(self, nt):
        return [(rhs,w) for (lhs,rhs,w) in self._nontermrules if lhs == nt]

    def term_expansions(self, nt):
        return [(rhs,w) for (lhs,rhs,w) in self._termrules if lhs == nt]

    ###################################################################################

    def nonterm_expectation(self, nt):
        return self._id[nt] * self._od[nt]

    def nonterm_rule_expectation(self, lhs, rhs, w):
        return self._od[lhs] * w * product(self._id[nt] for nt in rhs)

    def term_rule_expectation(self, lhs, rhs, w):
        return self._od[lhs] * w

    ###################################################################################

    def show_unary_graph(self, name=None):

        g = graphviz.Digraph(name)
        for (lhs,rhs,w) in self._nontermrules:
            if len(rhs) == 1 and w != 0:
                g.edge(lhs,rhs[0])
        filename = g.render()
        return filename

    def show_nullable_graph(self, name=None):

        g = graphviz.Digraph(name)
        # nullables = set([nt for (nt,rhs,w) in self._termrules if rhs is None and w != 0])
        # for nt in nullables:
        #     g.node(nt, style="filled")
        nullables = set([])
        edges = set([])
        for (lhs,rhs,wt) in self._termrules:
            if rhs is None and wt != 0:
                nullables.add(lhs)
                g.node(lhs, label=("%s %f" % (lhs,wt)), style="filled")
        while True:
            changed = False
            for (lhs,rhs,w) in self._nontermrules:
                if all(nt in nullables for nt in rhs):
                    if lhs not in nullables:
                        nullables.add(lhs)
                        g.node(lhs)
                        changed = True
                    for nt in rhs:
                        if (lhs,nt) not in edges:
                            edges.add((lhs,nt))
                            g.edge(lhs,nt)
                            changed = True
            if not changed:
                break
        filename = g.render()
        return filename

    # Returns a list of all nullable nonterminals, topologically sorted from ``lowest'' to highest
    def sorted_nullables(self):
        nullables = set([nt for (nt,rhs,w) in self._termrules if rhs is None and w != 0])
        edges = set([])
        while True:
            changed = False
            for (lhs,rhs,w) in self._nontermrules:
                if all(nt in nullables for nt in rhs):
                    if lhs not in nullables:
                        nullables.add(lhs)
                        changed = True
                    for nt in rhs:
                        if (lhs,nt) not in edges:
                            edges.add((lhs,nt))
                            changed = True
            if not changed:
                break
        finished_list = []
        dfscb(nullables, edges, cb_finished = lambda v,t: finished_list.append(v))
        return finished_list

    def remove_unary_loops(self):

        result_termrules = []
        result_nontermrules = []

        # Construct a directed graph representing the unary rewrite rules
        vertices = set([])
        edges = set([])
        for (lhs,rhs,w) in self._nontermrules:
            if len(rhs) == 1 and w != 0:
                vertices.add(lhs)
                vertices.add(rhs[0])
                edges.add((lhs,rhs[0]))

        components = strongly_connected_components(vertices, edges)

        for c in components:

            assert len(c) > 0

            if len(c) == 1:

                [nt] = c
                result_termrules += [(lhs,rhs,wt) for (lhs,rhs,wt) in self._termrules if lhs == nt]
                result_nontermrules += [(lhs,rhs,wt) for (lhs,rhs,wt) in self._nontermrules if lhs == nt]

            else:

                rule_weights = defaultdict(lambda: 0)
                for (lhs,rhs,w) in self._nontermrules:
                    if len(rhs) == 1:
                        [rhsnt] = rhs
                        if lhs in c and rhsnt in c:
                            rule_weights[(lhs,rhsnt)] += w

                dist = shortest_paths(c, rule_weights)

                tweaked = lambda s: s + "|"

                # Now put all the rules for each lhs v into the result grammar
                for v in c:

                    # terminal rules; one copy of each for v and for tweaked(v)
                    for (lhs,rhs,wt) in self._termrules:
                        if lhs == v:
                            result_termrules += [(lhs,rhs,wt), (tweaked(lhs),rhs,wt)]

                    # nonterminal rules not going to this component; one copy of each for v and for tweaked(v)
                    for (lhs,rhs,wt) in self._nontermrules:
                        if lhs == v:
                            if not (len(rhs) == 1 and rhs[0] in c):
                                result_nontermrules += [(lhs,rhs,wt), (tweaked(lhs),rhs,wt)]

                    # for each other vertex w in the component, include a rule from tweaked(v) to w
                    for w in c:
                        if w != v:
                            result_nontermrules += [(v, [tweaked(w)], dist[(v,w)])]

        return WeightedCFG(result_nontermrules, result_termrules, self._start_symbol)

    def remove_useless_rules(self):
        self._nontermrules = list(filter(lambda r: self.nonterm_rule_expectation(*r) != 0, self._nontermrules))
        self._termrules = list(filter(lambda r: self.term_rule_expectation(*r) != 0, self._termrules))

    # Eliminates epsilon rules if possible, or lifts them upwards so that the only epsilon rule is for the start symbol
    def raise_epsilon_rules(self):
        badnts = [lhs for (lhs,rhs,w) in self._termrules if rhs is None and lhs != self._start_symbol]
        print("badnts:", badnts)
        if badnts == []:
            return self
        else:
            print("Removing epsilon rule for", badnts[0])
            return self.remove_epsilon_rule(badnts[0]).raise_epsilon_rules()

    # Removes a particular epsilon rule 'X -> None', where X is not the start symbol. 
    # The process might introduce additional epsilon rules, but applying it repeatedly 
    # can get to a point where the only remaining epsilon rule (if any) is for the start symbol.
    def remove_epsilon_rule(self, nt):

        assert nt != self._start_symbol, "Can't remove an epsilon rule for the start symbol!"

        # Find the epsilon rule we're going to remove
        to_remove = lambda lhs,rhs,w: lhs == nt and rhs is None
        rules_to_remove = list(filter(lambda r: to_remove(*r), self._termrules))
        assert len(rules_to_remove) == 1
        [(_, _, epswt)] = rules_to_remove

        new_nontermrules = []
        new_termrules = list(filter(lambda r: not (to_remove(*r)), self._termrules))

        for (lhs,rhs,w) in self._nontermrules:
            # e.g. if we're eliminating 'X -> None' and this is the rule 'Y -> A X X', we 
            # collect up the variants [A,X,X], [A,X,None], [A,None,X], [A,None,None]
            variants = [[]]
            for x in rhs:
                if x == nt:
                    variants = [v+[x] for v in variants] + [v+[None] for v in variants]
                else:
                    variants = [v+[x] for v in variants]
            for v in variants:
                new_w = w * (epswt ** v.count(None))
                new_rhs = list(filter(lambda x: x is not None, v))
                if new_rhs != []:
                    new_nontermrules.append((lhs, new_rhs, new_w))
                else:
                    new_termrules.append((lhs, None, new_w))

        return WeightedCFG(new_nontermrules, new_termrules, self._start_symbol)

    def is_globally_normalized(self):

        return (round(self._id[self._start_symbol], 6) == 1)

    def globally_normalize(self):

        nontermrules = []
        termrules = []

        for (lhs,rhs,w) in self._nontermrules:
            if lhs == self._start_symbol:
                new_w = w / self._id[self._start_symbol]
            else:
                new_w = w
            nontermrules.append((lhs, rhs, new_w))

        for (lhs,rhs,w) in self._termrules:
            if lhs == self._start_symbol:
                new_w = w / self._id[self._start_symbol]
            else:
                new_w = w
            termrules.append((lhs, rhs, new_w))

        return WeightedCFG(nontermrules, termrules, self._start_symbol)

    def bottom_up_normalize(self):

        # Do global normalization first if necessary
        if not self.is_globally_normalized():
            return self.globally_normalize().bottom_up_normalize()

        nontermrules = []
        termrules = []

        for (lhs,rhs,w) in self._nontermrules:
            new_w = w * self._od[lhs] / product(self._od[nt] for nt in rhs)
            nontermrules.append((lhs, rhs, new_w))

        for (lhs,rhs,w) in self._termrules:
            new_w = w * self._od[lhs]
            termrules.append((lhs, rhs, new_w))

        return WeightedCFG(nontermrules, termrules, self._start_symbol)

    def is_top_down_normalized(self):

        return all(round(self._id[nt],6) == 1 for nt in self.all_nonterminals())

    def top_down_normalize(self):

        nontermrules = []
        termrules = []

        for (lhs,rhs,w) in self._nontermrules:
            # new_w = w * product(self._id[nt] for nt in rhs) / self._id[lhs]
            num = product(self._id[nt] for nt in rhs)
            denom = self._id[lhs]
            if num == 0:
                pass # Leave out this rule, any probability mass given to it will be thrown away
            else:
                nontermrules.append((lhs, rhs, w*num/denom))

        for (lhs,rhs,w) in self._termrules:
            new_w = w / self._id[lhs]
            termrules.append((lhs, rhs, new_w))

        return WeightedCFG(nontermrules, termrules, self._start_symbol)

    def derivative(self, nt):

        # We don't allow epsilon rules for anything other than the start symbol. Since the start symbol cannot occur 
        # on the right-hand side of any rule, this ensures that for a rule like 'A -> B C' we only need to 
        # introduce 'A_X -> B_X C', not 'A_X -> C_X', since B cannot produce the empty string.
        rule_ok = lambda lhs,rhs,w: rhs is not None or lhs == self._start_symbol
        assert(all(rule_ok(*r) for r in self._termrules)), "Grammar for derivative must not contain non-initial epsilon rules"

        new_nontermrules = []
        new_termrules = []

        for (lhs,rhs,w) in self._nontermrules:

            new_lhs = "%s_%s" % (lhs,nt)
            new_rhs0 = "%s_%s" % (rhs[0],nt)
            new_nontermrules.append((new_lhs, [new_rhs0] + rhs[1:], w))

            if rhs[0] == nt:
                if len(rhs) > 1:
                    new_nontermrules.append((new_lhs, rhs[1:], w))
                else:
                    new_termrules.append((new_lhs, None, w))

        new_start_symbol = "%s_%s" % (self._start_symbol,nt)

        g = WeightedCFG(new_nontermrules + self._nontermrules, new_termrules + self._termrules, new_start_symbol)
        return g

    # The argument stack_segments is a list of pairs of the form (corners,prediction), where corners is a 
    # list of nonterminals that have been found bottom-up, and prediction is a single nonterminal that those 
    # found constituents are initial portions of.
    # For example after a sentence-initial D triggers the rule 'NP -> D N', we can imagine passing [([],"N"), (["NP"],"S")].
    def stack_derivative(self, stack_segments):

        assert stack_segments != []

        components = []

        for (found_corners,prediction) in stack_segments:
            d = self
            start_symbol = prediction
            for nt in found_corners:
                d = d.raise_epsilon_rules()
                d = d.derivative(nt)
                start_symbol += "_" + nt
            d = d.top_down_normalize()
            components.append((d, start_symbol))

        g = WeightedCFG.concat_grammars(components)
        return g

    def first_word_probs(self):

        # If we have any epsilon rules for nonterminals other than the start symbol, fix that first
        if any((rhs is None and lhs != self._start_symbol) for (lhs,rhs,w) in self._termrules):
            return self.raise_epsilon_rules().top_down_normalize().first_word_probs()

        assert self.is_top_down_normalized()

        # The algorithm here is based on the presentation of the Jelinek-Lafferty algorithm in 
        # Nowak & Cotterell 2023, ``A Fast Algorithm for Computing Prefix Probabilities'', pp.59-60. 
        # Since we only care about the first word, we only care about their base case.

        # Collect up the entries for the matrix that Nowak & Cotterell call P
        lc_dict = defaultdict(lambda: 0)
        for (lhs,rhs,w) in self._nontermrules:
            lc_dict[(lhs,rhs[0])] += w

        # Construct the matrix P, calculate P^\star, and pull out the row corresponding to the start symbol
        nts = self.all_nonterminals()
        matrix_p = np.array([[lc_dict.get((x,y),0) for y in nts] for x in nts])
        matrix_pstar = np.linalg.inv(np.identity(len(nts)) - matrix_p)
        [pstar_startsym_row] = [row for (nt,row) in zip(nts,matrix_pstar) if nt == self._start_symbol]
        pstar_startsym_dict = {nt:p for (nt,p) in zip(nts,pstar_startsym_row)}

        # Put together E_lc(...|ROOT) with the terminal rules, following N&C's equation (14)
        result_dict = defaultdict(lambda:0)
        for (lhs,rhs,w) in self._termrules:
            if rhs is None:
                assert lhs == self._start_symbol, "Found epsilon rule for %s, which is not the start symbol" % lhs
            result_dict[rhs] += pstar_startsym_dict[lhs] * w

        return result_dict

    def first_word_check(self,n=10000):

        # If we have any epsilon rules for nonterminals other than the start symbol, fix that first
        if any((rhs is None and lhs != self._start_symbol) for (lhs,rhs,w) in self._termrules):
            self.raise_epsilon_rules().top_down_normalize().first_word_check()
            return

        assert self.is_top_down_normalized()

        def first_word(tree):
            return tree_fold(None, lambda s: s, lambda subresults: subresults[0], tree)

        def yld(tree):
            return tree_fold([], lambda s: [s], lambda subresults: [w for ws in subresults for w in ws], tree)

        def tree_fold(eps, f_term, f_daughters, tree):
            (parent,contents) = tree
            if contents is None:
                return eps
            elif type(contents) is str:
                return f_term(contents)
            else:
                assert (type(contents) is list and len(contents) > 0), ("contents is: %s" % str(contents))
                return f_daughters([tree_fold(eps, f_term, f_daughters, d) for d in contents])

        c1 = Counter()
        c2 = Counter()
        for _ in range(n):
            t = self.sample_tree()
            c1[first_word(t)] += 1
            c2[tuple(yld(t)[:1])] += 1

        top_words = sorted(self.first_word_probs().items(), key=lambda p:p[1], reverse=True)

        triples_of_pairs = zip(top_words, c1.most_common(), c2.most_common())
        rows = [pair1 + pair2 + pair3 for (pair1,pair2,pair3) in itertools.islice(triples_of_pairs,20)]
        print(tabulate(rows, tablefmt="plain", floatfmt=".6f"))

    def chart_parse(self, symbols):

        chart = defaultdict(lambda: [])

        for length in range(1, len(symbols)+1):

            for startpos in range(0, len(symbols) - length + 1):

                substr = tuple(symbols[startpos : startpos + length])

                if length == 1:
                    for (lhs,rhs,wt) in self._termrules:
                        if (rhs,) == substr:
                            new_nt = (lhs, startpos, startpos+length)
                            chart[new_nt] += [(rhs,wt)]
                else:
                    for (lhs,(rhs1,rhs2),wt) in self._nontermrules:
                        for k in range(1,length):
                            new_rhs1 = (rhs1, startpos, startpos+k)
                            new_rhs2 = (rhs2, startpos+k, startpos+length)
                            if (new_rhs1 in chart) and (new_rhs2 in chart):
                                chart[(lhs, startpos, startpos + length)] += [((new_rhs1,new_rhs2),wt)]

        return chart

    def intersect(self, symbols):

        chart = self.chart_parse(symbols)

        nontermrules = []
        termrules = []

        for ((nt,i,j), rhss) in chart.items():
            if i+1 == j:
                for (rhs,wt) in rhss:
                    termrules.append(((nt,i,j), rhs, wt))
            else:
                for (rhs,wt) in rhss:
                    nontermrules.append(((nt,i,j), rhs, wt))

        return WeightedCFG(nontermrules, termrules, (self._start_symbol,0,len(symbols)))

    def sample_tree(self, root=None):

        if root is None:
            root = self._start_symbol

        unzip = lambda x: tuple(zip(*x))    # magic hack!
        nonterm_options = self.nonterm_expansions(root)
        term_options = self.term_expansions(root)
        (options,weights) = unzip(nonterm_options + term_options)

        [chosen_rhs] = random.choices(options, weights)

        # If the chosen RHS is either None (i.e. an epsilon rule) or a string (i.e. a terminal)
        if chosen_rhs is None or type(chosen_rhs) is str:
            return (root, chosen_rhs)
        else:
            daughters = [self.sample_tree(nt) for nt in chosen_rhs]
            return (root, daughters)

    def random_tree(self, root=None):

        if root is None:
            root = self._start_symbol

        nonterm_options = self.nonterm_expansions(root)
        term_options = self.term_expansions(root)

        [rewrite_as_terminal] = random.choices([True, False], weights=[len(term_options), len(nonterm_options)])

        if rewrite_as_terminal:

            (rhs,wt) = random.choice(term_options)
            return (wt, (root,rhs))

        else:

            (rhs,wt) = random.choice(nonterm_options)
            result_weight = wt
            result_daughters = []
            for nt in rhs:
                (w,daughter) = self.random_tree(nt)
                result_weight *= w
                result_daughters.append(daughter)
            result_tree = (root, result_daughters)
            return (result_weight, result_tree)

    def report(self):

        headers = ["nonterminal", "inside", "outside", "expectation"]
        rows = []
        for nt in self.all_nonterminals():
            rows.append([nt, self._id[nt], self._od[nt], self.nonterm_expectation(nt)])

        # If the grammar is top-down normalized (all inside values are 1), do sampling
        if self.is_top_down_normalized():

            # Find frequencies of all symbols in a sample of trees
            total_frequencies = Counter()
            total_trees = 10000
            try:
                for x in range(total_trees):
                    t = self.sample_tree()   # Assumes that the grammar is locally-normalized and consistent
                    total_frequencies.update(symbols_in_tree(t))
            except RecursionError as e:
                print("*** Recursion error, couldn't get samples (%s)" % e, file=sys.stderr)
            else:
                headers.append("from sample")
                for (nt,row) in zip(self.all_nonterminals(), rows):
                    row.append(total_frequencies[nt]/total_trees)

        print()
        print(tabulate(rows, headers=headers, tablefmt="plain", floatfmt=".6f"))

        # print()
        # print("Outside values from equation-solver:")
        # print(self._od2)

        headers = ["weight", "rule", "expectation"]
        rows = []
        for (lhs,rhs,w) in self._nontermrules:
            rule_string = "%s --> %s" % (lhs, " ".join(map(str,rhs)))
            rows.append([w, rule_string, self.nonterm_rule_expectation(lhs,rhs,w)])
        for (lhs,rhs,w) in self._termrules:
            rule_string = "%s --> %s" % (lhs, rhs)
            rows.append([w, rule_string, self.term_rule_expectation(lhs,rhs,w)])
        print()
        print(tabulate(rows, headers=headers, tablefmt="plain", floatfmt=".6f"))

        # # Now try to get the expectations in a way that does not assume local-normalization
        # weighted_frequencies = defaultdict(lambda: 0)
        # total_weight = 0
        # for x in range(1000):
        #     t = None
        #     while t is None:
        #         try:
        #             (w,t) = self.random_tree()     # Samples in a way that does not use weights
        #         except RecursionError:
        #             pass
        #     total_weight += w
        #     symbol_counter = symbols_in_tree(t)
        #     for (symbol,freq) in symbol_counter.items():
        #         weighted_frequencies[symbol] += w*freq
        # print("\nWeighted frequencies:")
        # for (symbol,wf) in weighted_frequencies.items():
        #     print("%8s\t%f\t%f" % (symbol, wf, wf/total_weight))

    def show_samples(self, n=20):

        if self.is_top_down_normalized():
            for i in range(n):
                print(self.sample_tree())
        else:
            print("Can't sample, grammar is not top-down normalized")

############################################################################################

def symbols_in_tree(tree):
    c = Counter()
    (root_symbol, rest) = tree
    c[root_symbol] += 1
    if type(rest) is str or rest is None:  # None for epsilon rules
        c[rest] += 1
    else:
        for t in rest:
            c.update(symbols_in_tree(t))
    return c

############################################################################################

def read_grammar_from_file(filename):
    f = open(filename)
    g = read_grammar(f)
    f.close()
    return g

def read_grammar(f):
    nontermrules = []
    termrules = []
    for line in f:
        line = line.rstrip()
        if line.startswith("#"):
            continue
        m0 = re.fullmatch("(\d+)\s*/\s*(\d+)\s*([\w\-]+)\s* --> \s*(.*)", line)
        m1 = re.fullmatch("\(\* .* \*\)", line)
        if m0 is not None:
            (num, denom, lhs, rhs) = m0.groups()
            weight = int(num)/int(denom)
            mrhs1 = re.fullmatch("\"([\w\-]*)\"", rhs)
            mrhs2 = re.fullmatch("((?:[\w\-]+\s+)*[\w\-]+)\s*((?:\[[\d,;]+\])*)", rhs)       # (?:...) is a non-capturing group
            if mrhs1 is not None:
                (terminal,) = mrhs1.groups()
                if terminal == "":
                    terminal = None
                termrules.append((lhs, terminal, weight))
            elif mrhs2 is not None:
                (nonterms,recipe) = mrhs2.groups()
                nontermrules.append((lhs, nonterms.split(), weight))
            else:
                print("WARNING 2: Ignoring ill-formed grammar line: %s" % line, file=sys.stderr)
        elif m1 is not None:
            pass
        else:
            print("WARNING 1: Ignoring ill-formed grammar line: %s" % line, file=sys.stderr)
    return WeightedCFG(nontermrules, termrules)

############################################################################################

def main():
    pass

if __name__ == "__main__" and not sys.flags.interactive:
    main()

    # TODO Ideas:
    #       Include ``daughters given parent'' and ``parent and sisters given daughter'' probabilities in report.

