import numpy as np
import rpxdock as rp

def score_fun2(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.mean(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.mean(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      side1_stdev = np.std(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2_stdev = np.std(ressc2[lbub1[i, 0]:lbub1[i, 1]])
         # TODO: maybe do this a different way?
      mscore = (side1 + side2) / 2
      sscore = np.sqrt(side1_stdev**2 + side2_stdev**2)
      if np.isnan(sscore):
         sscore = 100

   #ncont_score = a * np.exp( -((ncont) - b)**2 / (2*c**2) )
      a=300
      mu = 75
      sigma = 50
      b=np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
      c=np.log(1 + (sigma**2 / mu**2))
      #ncont_score = a * np.exp( -((ncont) - mu)**2 / (2*sigma**2) )
      if (ub - lb) > 0:
         ncont_score = ( a  / (c * np.sqrt(2 * np.pi) * (ub - lb)) ) * np.exp( -(np.log(ub - lb) - b)**2 / (2*c**2) )
         scores[i] = kw.wts.rpx * mscore + ncont_score
      else:
         scores[i] = 0
   return scores
   
def sasa_priority(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))

   #TODO: Quinton: Resolution-dependent scoring is turned off while I try to optimize it further. This does nothing right now.
   if kw.wts.ncontact != 0:
      if kw.wts.rpx != 0:
         #start_sasa = kw.wts.sasa + ( 576 * 4 )
         #sasa = start_sasa - ( 576 * kw.iresl )
         sasa = kw.wts.sasa
         #if not kw.wts.error:
         #   start_error = 6
         #else:
         #   start_error = kw.wts.error + 2
         #sigma = start_error - ( kw.iresl / 2 )
         sigma = kw.wts.error
      else:
         if not kw.wts.error:
            sigma = 4
         else:
            sigma = kw.wts.error
         sasa = kw.wts.sasa
   else:
      if not kw.wts.error:
          sigma = 4
      else:
          sigma = kw.wts.error
      sasa = kw.wts.sasa
  
   #calculate constants based on weightings
   a = kw.wts.ncontact
   m = np.exp((-sigma**2.22215285) / 28.59075188)
   mu = (sasa) / m #convert input sasa (mode) into a mean
   mu = (mu - 14.2198) / 21.522 #redefine mean in terms of ncontact
   sigma = mu * sigma * 0.2433619617913417 #redefine mean in terms of ncontact
   mode = (sasa - 14.2198) / 21.522 #redefine mode in terms of ncontact

   #calculate parameterization factors
   b=np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
   c=np.log(1 + (sigma**2 / mu**2))

   #normalization of the lognormal distribution to the maximum score so that all possible sasa/sigma combinations
   #result in the same maximum possible score
   prob_max = ( 1  / (c * np.sqrt(2 * np.pi) * (mode)) ) * np.exp( -(np.log(mode) - b)**2 / (2*c**2) )

   #score docks
   for i, (lb, ub) in enumerate(lbub):
      if (ub - lb) > 0:
         side1 = np.mean(ressc1[lbub1[i, 0]:lbub1[i, 1]])
         side2 = np.mean(ressc2[lbub2[i, 0]:lbub2[i, 1]])
         # TODO: maybe do this a different way?
         mscore = (side1 + side2) / 2
      
         #ncont_score
         ncont_score = ( a / prob_max )*( 1  / (c * np.sqrt(2 * np.pi) * (ub - lb)) ) * np.exp( -(np.log(ub - lb) - b)**2 / (2*c**2) )
         
         scores[i] = kw.wts.rpx * mscore + ncont_score
      else:
         
         scores[i] = 0
   return scores

def stnd(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.sum(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.sum(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      mscore = (side1 + side2) / 2
      scores[i] = kw.wts.rpx * mscore + kw.wts.ncontact * (ub - lb)
   return scores
  
def mean(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.mean(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.mean(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      mscore = (side1 + side2) / 2
      scores[i] = kw.wts.rpx * mscore + kw.wts.ncontact * (ub - lb)
   return scores
  
def median(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.median(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.median(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      mscore = (side1 + side2) / 2
      scores[i] = kw.wts.rpx * mscore + kw.wts.ncontact * (ub - lb)
   return scores
  
def exp(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.sum(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.sum(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      mscore = (side1 + side2) / 2
      scores[i] = mscore - ( 4.6679 * ((ub - lb)**0.588  ))
   return scores
  
def lin(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.sum(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.sum(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      mscore = (side1 + side2) / 2
      scores[i] = mscore - ( 0.7514*(ub - lb) )
   return scores
