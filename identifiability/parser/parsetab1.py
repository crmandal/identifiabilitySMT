
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'nonassocLTGTLEGEleftADDSUBleftMULTDIVleftUMINUSrightPOWADD AND AP AT CLN CM COMMENT COS DEFN DF DIV EQ EXP F GE GT LB LC LE LOG LP LT MULT NUM POW RAT RB RC RP SIN SQRT SUB T TAN TO VARequations : equations equationequations : equationequation : VAR EQ exprexpr : expr ADD expr\n\t| expr SUB expr\n\t| expr MULT expr\n\t| expr DIV expr\n\t| expr POW exprexpr : LP expr RPexpr : LP SUB expr RP %prec UMINUSexpr : rangeexpr : trig_func\n\t\t | exp_functrig_func : trig LP expr RPexp_func : EXP LP expr RP \n\t\t\t\t| LOG LP expr RP\n\t\t\t\t| SQRT LP expr RPtrig : SIN\n\t\t\t| COS\n\t\t\t| TANempty :range : NUM\n\t\t\t| RAT\n\t\t\t| VAR'
    
_lr_action_items = {'VAR':([0,1,2,4,5,6,7,8,9,10,11,12,13,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,43,44,45,46,47,],[3,3,-2,-1,6,-24,-3,6,-11,-12,-13,-22,-23,6,6,6,6,6,6,6,6,6,6,-4,-5,-6,-7,-8,-9,-10,-14,-15,-16,-17,]),'$end':([1,2,4,6,7,9,10,11,12,13,32,33,34,35,36,37,43,44,45,46,47,],[0,-2,-1,-24,-3,-11,-12,-13,-22,-23,-4,-5,-6,-7,-8,-9,-10,-14,-15,-16,-17,]),'EQ':([3,],[5,]),'LP':([5,8,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,],[8,8,28,29,30,31,-18,-19,-20,8,8,8,8,8,8,8,8,8,8,]),'NUM':([5,8,21,22,23,24,25,27,28,29,30,31,],[12,12,12,12,12,12,12,12,12,12,12,12,]),'RAT':([5,8,21,22,23,24,25,27,28,29,30,31,],[13,13,13,13,13,13,13,13,13,13,13,13,]),'EXP':([5,8,21,22,23,24,25,27,28,29,30,31,],[15,15,15,15,15,15,15,15,15,15,15,15,]),'LOG':([5,8,21,22,23,24,25,27,28,29,30,31,],[16,16,16,16,16,16,16,16,16,16,16,16,]),'SQRT':([5,8,21,22,23,24,25,27,28,29,30,31,],[17,17,17,17,17,17,17,17,17,17,17,17,]),'SIN':([5,8,21,22,23,24,25,27,28,29,30,31,],[18,18,18,18,18,18,18,18,18,18,18,18,]),'COS':([5,8,21,22,23,24,25,27,28,29,30,31,],[19,19,19,19,19,19,19,19,19,19,19,19,]),'TAN':([5,8,21,22,23,24,25,27,28,29,30,31,],[20,20,20,20,20,20,20,20,20,20,20,20,]),'ADD':([6,7,9,10,11,12,13,26,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,],[-24,21,-11,-12,-13,-22,-23,21,-4,-5,-6,-7,-8,-9,21,21,21,21,21,-10,-14,-15,-16,-17,]),'SUB':([6,7,8,9,10,11,12,13,26,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,],[-24,22,27,-11,-12,-13,-22,-23,22,-4,-5,-6,-7,-8,-9,22,22,22,22,22,-10,-14,-15,-16,-17,]),'MULT':([6,7,9,10,11,12,13,26,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,],[-24,23,-11,-12,-13,-22,-23,23,23,23,-6,-7,-8,-9,23,23,23,23,23,-10,-14,-15,-16,-17,]),'DIV':([6,7,9,10,11,12,13,26,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,],[-24,24,-11,-12,-13,-22,-23,24,24,24,-6,-7,-8,-9,24,24,24,24,24,-10,-14,-15,-16,-17,]),'POW':([6,7,9,10,11,12,13,26,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,],[-24,25,-11,-12,-13,-22,-23,25,25,25,25,25,25,-9,25,25,25,25,25,-10,-14,-15,-16,-17,]),'RP':([6,9,10,11,12,13,26,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,],[-24,-11,-12,-13,-22,-23,37,-4,-5,-6,-7,-8,-9,43,44,45,46,47,-10,-14,-15,-16,-17,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'equations':([0,],[1,]),'equation':([0,1,],[2,4,]),'expr':([5,8,21,22,23,24,25,27,28,29,30,31,],[7,26,32,33,34,35,36,38,39,40,41,42,]),'range':([5,8,21,22,23,24,25,27,28,29,30,31,],[9,9,9,9,9,9,9,9,9,9,9,9,]),'trig_func':([5,8,21,22,23,24,25,27,28,29,30,31,],[10,10,10,10,10,10,10,10,10,10,10,10,]),'exp_func':([5,8,21,22,23,24,25,27,28,29,30,31,],[11,11,11,11,11,11,11,11,11,11,11,11,]),'trig':([5,8,21,22,23,24,25,27,28,29,30,31,],[14,14,14,14,14,14,14,14,14,14,14,14,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> equations","S'",1,None,None,None),
  ('equations -> equations equation','equations',2,'p_equations','parseEquations.py',111),
  ('equations -> equation','equations',1,'p_equations1','parseEquations.py',116),
  ('equation -> VAR EQ expr','equation',3,'p_equation','parseEquations.py',121),
  ('expr -> expr ADD expr','expr',3,'p_exp1','parseEquations.py',128),
  ('expr -> expr SUB expr','expr',3,'p_exp1','parseEquations.py',129),
  ('expr -> expr MULT expr','expr',3,'p_exp1','parseEquations.py',130),
  ('expr -> expr DIV expr','expr',3,'p_exp1','parseEquations.py',131),
  ('expr -> expr POW expr','expr',3,'p_exp1','parseEquations.py',132),
  ('expr -> LP expr RP','expr',3,'p_exp5','parseEquations.py',139),
  ('expr -> LP SUB expr RP','expr',4,'p_exp2','parseEquations.py',146),
  ('expr -> range','expr',1,'p_exp3','parseEquations.py',152),
  ('expr -> trig_func','expr',1,'p_exp4','parseEquations.py',156),
  ('expr -> exp_func','expr',1,'p_exp4','parseEquations.py',157),
  ('trig_func -> trig LP expr RP','trig_func',4,'p_trig_func','parseEquations.py',161),
  ('exp_func -> EXP LP expr RP','exp_func',4,'p_exp_func','parseEquations.py',167),
  ('exp_func -> LOG LP expr RP','exp_func',4,'p_exp_func','parseEquations.py',168),
  ('exp_func -> SQRT LP expr RP','exp_func',4,'p_exp_func','parseEquations.py',169),
  ('trig -> SIN','trig',1,'p_trig','parseEquations.py',175),
  ('trig -> COS','trig',1,'p_trig','parseEquations.py',176),
  ('trig -> TAN','trig',1,'p_trig','parseEquations.py',177),
  ('empty -> <empty>','empty',0,'p_empty','parseEquations.py',181),
  ('range -> NUM','range',1,'p_range','parseEquations.py',185),
  ('range -> RAT','range',1,'p_range','parseEquations.py',186),
  ('range -> VAR','range',1,'p_range','parseEquations.py',187),
]
