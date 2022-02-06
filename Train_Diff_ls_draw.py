#ls 相关实验绘图
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
label_ls=np.linspace(start=11,stop=20,num=10,dtype=int)
label_ls=np.linspace(start=0.0,stop=80,num=17)*0.01
print(label_ls)

#ls
# # targeted
# clean=np.array([98.17,97.88,98.33,98.20,98.72,98.76,97.84,97.91,98.46,98.34,98.57,98.98,98.42,98.12,99.16,98.15,97.79])
# before_fgsm=np.array([16.65,19.53,28.51,19.26,12.31,8.84,6.43,11.98,14.94,31.33,18.26,2.69,18.84,9.01,6.94,13.19,5.87])
# after_fgsm = np.array([92.48,75.34,86.53,91.36,97.98,96.28,95.59,95.86,96.56,82.58,94.12,97.53,86.85,92.07,99.49,89.89,95.43])
# before_pgd = np.array([32.22,31.78,53.18,33.34,52.06,46.8,52.57,66.77,52.72,55.31,45.65,53.56,34.04,35.77,27.16,15.64,17.37])
# after_pgd = np.array([32.64,32.58,55.95,36.42,55.91,55.87,65.52,72.57,64.32,57.94,61.33,68.61,53.98,58.94,54.46,28.51,64.75])

# untargeted
clean = np.array([98.17,97.88,98.33,98.20,98.72,98.76,97.84,97.91,98.46,98.36,98.57,98.98,98.42,98.12,99.16,98.15,97.39])
before_fgsm = np.array([6.41,16.71,52.68,39.70,33.50,41.34,54.80,38.03,26.07,34.75,41.41,24.62,42.59,41.95,13.27,8.67,18.18])
after_fgsm = np.array([89.95,91.01,90.60,90.80,96.31,95.50,94.19,93.40,95.26,87.28,95.67,98.35,89.78,89.97,99.57,94.65,91.23])
before_pgd = np.array([0.95,3.00,16.66,4.39,7.18,7.26,20.59,14.00,5.00,15.22,14.50,10.22,21.13,15.07,0.12,2.57,6.68])
after_pgd = np.array([91.9,90.23,90.35,90.97,97.15,97.52,95.37,94.42,97.19,83.75,95.71,97.66,82.07,85.49,99.89,85.97,85.00])

# #输出个数
# # untargeted
# clean=np.array([99.55,99.27,98.43,98.49,98.38,98.23,98.34,98.24,97.99,97.91])
# before_fgsm=np.array([63.17,14.53,53.96,66.77,41.60,18.24,28.00,43.17,13.75,18.40])
# after_fgsm = np.array([91.71,98.33,79.90,84.49,86.11,92.13,87.61,93.70,96.55,84.10])
# before_pgd = np.array([20.47,0.70,16.16,10.65,10.05,0.84,3.99,1.51,0.58,3.20])
# after_pgd = np.array([59.26,96.11,59.38,70.94,80.59,95.78,86.03,92.03,98.14,88.64])
#
# # targeted
# clean=np.array([99.55,99.27,98.43,98.49,98.38,98.23,98.34,98.24,97.99,97.91])
# before_fgsm=np.array([31.89,10.03,31.43,37.49,26.68,19.91,15.88,17.65,12.54,11.59])
# after_fgsm = np.array([80.74,95.72,68.27,74.13,78.04,91.20,92.06,90.85,95.01,96.16])
# before_pgd = np.array([12.51,11.58,28.72,37.35,26.72,27.80,21.71,26.94,26.67,35.39])
# after_pgd = np.array([82.59,55.35,43.37,67.52,32.12,36.54,29.26,31.71,29.39,41.92])

lw=4
plt.plot(label_ls, clean,"b-d",
         lw=lw, label='%s ' % ('clean'),ms=10)
plt.plot(label_ls, before_fgsm,"g--o",
         lw=lw, label='%s' % ('fgsm--before detecting'),ms=10)
plt.plot(label_ls, after_fgsm,"g--v",
         lw=lw+2, label='%s ' % ('fgsm--after detecting'),ms=10)
plt.plot(label_ls, before_pgd,"r-.o",
         lw=lw, label='%s ' % ('pgd--before detecting'),ms=10)
plt.plot(label_ls, after_pgd,"r-.v",
         lw=lw+2, label='%s ' % ('pgd--after detecting'),ms=10)

# plt.plot(fpr_dict["micro"], tpr_dict["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc_dict["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr_dict["macro"], tpr_dict["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc_dict["macro"]),
#          color='navy', linestyle=':', linewidth=4)


x_major_locator=MultipleLocator(0.05)
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# plt.xlim([11,20])
plt.title("The Impact of Ls Rate(Untargeted attack)",fontsize=25)
plt.ylim([0.0, 100])
plt.xlabel('Labelsmoothing Factor', fontsize=25)
plt.ylabel('accuracy', fontsize=25)
plt.legend(loc="best",fontsize=12)
plt.show()