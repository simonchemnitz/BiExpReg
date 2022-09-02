# BiExpReg
Bi-exponential regression, (two-phase decay)

Fits the model defined by

$SpanFast=(Y0-Plateau)\cdot PercentFast\cdot 0.01$

$SpanSlow=(Y0-Plateau)\cdot (100-PercentFast)\cdot 0.01$

$Y=Plateau + SpanFast\cdot\exp(-KFast\cdot X) + SpanSlow\cdot\exp(-KSlow\cdot X)$
