
import numpy as np

'''keeping species alive'''
#C0 = np.array([[4.4444444444], [2.222222222]]) # noise
#C0 = np.array([[10.],[1./3.]]) # normal
#C0 = np.array([[10.],[5.5555555555]]) # recurrent

#SS = np.array([[1.05879964],[1.94953122]]) # noise
#SS = np.array([[5.86086775], [2.98599383]]) # normal
#SS = np.array([[5.27121812],[5.117429]]) # reccurrent

'''finer control 4.5 > N > 3.5'''
#C0 = np.array([[5.555555556], [4.44444444]])
#SS = np.array([[4.37927702], [4.03026668]])


J = np.array([[-0.01 * SS[0], -0.06 * SS[0]],
              [-0.01 * SS[0], -0.01 * SS[0]]]).reshape(2,2)

print(np.linalg.eig(J))
