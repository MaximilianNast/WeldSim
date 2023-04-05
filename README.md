# WeldSim
This repository contains 3 of the Models discussed in a 2023 research paper titled "Modell zur Berechnung der
Schweißraupenhöhen beim mehrlagigen Drahtauftragschweißen zur Anwendung für die additive Fertigung (WAAM)", as well
as an in-code comparison of the results.

# Krinke_model_full:
The main model as proposed by Stefan Krinke, M. Sc. in the paper mentioned above. The models equation requires 24
coefficients. This model does not contain a training function.

# Krinke_model_reduced:
The reduced model is marginally reduced in performance but requires only 9 coefficients and includes a training
function.

# Rios_model:
A physics based model, proposed by Rios et. al in https://www.sciencedirect.com/science/article/pii/S2214860417303470