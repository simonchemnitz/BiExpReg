# BiExpReg
Bi-exponential regression, (two-phase decay)

Fits the model define by

$SpanFast=(Y0-Plateau)\cdotPercentFast\cdot0.01$

$SpanSlow=(Y0-Plateau)\cdot(100-PercentFast)\cdot0.01$

$Y=Plateau + SpanFast*exp(-KFast\cdotX) + SpanSlow*exp(-KSlow\cdotX)$
