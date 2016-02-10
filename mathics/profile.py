#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import cProfile
import pstats

from mathics.core.definitions import Definitions
from mathics.core.evaluation import Evaluation

definitions = Definitions(add_builtin=True)


def prepare():
    pass

result = None


def run():
    global result
    # prompt = '(1+a)(1+b)(1+c)(1+d)(1+e)//Expand'
    # prompt = 'f/@Range[20000];'
    # prompt = 'Plus @@ Range[50000]'
    # prompt = 'Range[100000];'
    try:
        # prompt = 'SetAttributes[v, Flat]; v[x_]:={x}; v[a,b]'
        # prompt = """(Plus@@Symbol/@CharacterRange["a","z"])^2//Expand;"""
        # prompt = (
        #     'Plus@@f/@Symbol/@StringJoin/@Tuples[CharacterRange["a","z"],2]')
        # prompt = 'FullForm[Nest[1+Sqrt[1+#]&, x, 20]]'
        # prompt = '1+2'
        prompt = 'DensityPlot[x*y,{x,-1,1},{y,-1,1}]'
        evaluation = Evaluation(prompt, definitions, format='xml')
        if evaluation.results:
            result = evaluation.results[0].result
    except KeyboardInterrupt:
        result = 'INTERRUPTED'


def _profile():
    global result
    prepare()
    cProfile.run('run()', 'profile')
    p = pstats.Stats('profile')
    p.sort_stats('cumulative').print_stats(50)
    p.print_callees(20)

if __name__ == '__main__':
    _profile()
