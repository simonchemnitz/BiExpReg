# BiExpReg
Bi-exponential regression, (two-phase decay)

Fits the model define by
$SpanFast=(Y0-Plateau)*PercentFast*.01$
$SpanSlow=(Y0-Plateau)*(100-PercentFast)*.01$
$Y=Plateau + SpanFast*exp(-KFast*X) + SpanSlow*exp(-KSlow*X)$
