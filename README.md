# WeldSim
This repository contains 3 of the Models discussed in a 2023 research paper titled "Modell zur Berechnung der
Schweißraupenhöhen beim mehrlagigen Drahtauftragschweißen zur Anwendung für die additive Fertigung (WAAM)", as well
as an in-code comparison of the results.

# Krinke_model_full:
The main model as proposed by Stefan Krinke, M. Sc. in the paper mentioned above. The models equation requires 24
coefficients. This model does not contain a training function. <br>
The "krinke_model_coefficients" directiory contains
the coefficients found after training the model on "output.xlsx". If you want to e.g. calculate the height of a single 
layer, use the coefficients stored in "single_layer_height.csv" and plug them into the equation defined in the 
`evaluate` function. <br>
If you want to calculate the height of multiple layers, you may use the `calculate_layer_height` function as shown in the
`__main__` method.

# Rios_model:
A physics based model, proposed by Rios et. al in https://www.sciencedirect.com/science/article/pii/S2214860417303470