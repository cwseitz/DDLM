import matplotlib.pyplot as plt

# Data
N0_values = [2,5,10]

xerr = [0.83,0.62,0.57]
txerr = [0.97,0.75,0.66]
crlb = [10,6,4]
precision = [ 0.9744689144416293, 1.0, 0.9960159362549801]
tprecision = [0.9942334460131239, 0.9912767644726408, 0.9988014382740711]

recall = [0.8424599831508003,0.8854258898530193, 0.8779631255487269]
trecall = [0.9036688957166095, 0.9164222873900293, 0.906289650172195]

xerr = [27 * x for x in xerr]
txerr = [27 * y for y in txerr]

fig, axs = plt.subplots(1,3,figsize=(6, 2))

axs[0].plot(N0_values, xerr, marker='o', color='black', label='Diffusion')
axs[0].plot(N0_values, txerr, marker='o', color='gray', label='DeepSTORM')
axs[0].plot(N0_values, crlb, marker='o', color='red', label='CRLB')
axs[0].set_xlabel('SNR')
axs[0].set_ylabel('Localization Error (nm)')
#axs[0].legend()

axs[1].plot(N0_values, precision, marker='o', color='black',label='Diffusion')
axs[1].plot(N0_values, tprecision, marker='o', color='gray', label='DeepSTORM')
axs[1].set_xlabel('SNR')
axs[1].set_ylabel('Precision')
axs[1].set_xticks(N0_values)
axs[1].set_ylim([0.7,1.1])

axs[2].plot(N0_values, recall, marker='o', color='black',label='Diffusion')
axs[2].plot(N0_values, trecall, marker='o', color='gray', label='DeepSTORM')
axs[2].set_xlabel('SNR')
axs[2].set_ylabel('Recall')
axs[2].set_xticks(N0_values)
axs[2].set_ylim([0.7,1.1])

plt.tight_layout()
plt.show()
