import numpy as np
import matplotlib.pyplot as plt


def frob_norm(x):
    N = 0
    for i in range(len(x)):
        N += (x[i])**2
    return np.sqrt(N)

# funkce jednotlivych metod
def prosta_m(A,b, iter):
    U = np.subtract(np.eye(len(b)), A)
    x = np.zeros(len(b))
    np.copyto(x,b)
    x_next = np.zeros(len(b))
    deviation = np.zeros(iter)
    l  = 0
    while l < iter:
        for i in range(len(b)):
            L = 0
            for j in range(len(b)):
                L += (U[i][j])*(x[j])
            x_next[i] = L + b[i]
        deviation[l] = frob_norm(np.subtract(x, x_next))
        np.copyto(x, x_next)
        l += 1
    return x, deviation

def gauss_seidel_m(A, b, iter):
    x = np.zeros(len(b))
    np.copyto(x,b)
    x_next = np.zeros(len(b))
    np.copyto(x_next,b)
    deviation = np.zeros(iter)
    l = 0
    while l < iter:
        for i in range(len(b)):
            L = 0
            for j in range(len(b)):
                if j != i:
                    L += A[i][j]*x_next[j]
            x_next[i] = (b[i]-L)/A[i][i]
        deviation[l] = frob_norm(np.subtract(x_next,x))
        print(x_next)
        np.copyto(x, x_next)
        l += 1
    return x, deviation
    

def jacobiho_m(A, b, iter):
    x = np.zeros(len(b))
    np.copyto(x,b)
    x_next = np.zeros(len(b))
    deviation = np.zeros(iter)
    l = 0
    while l < iter:
        for i in range(len(b)): # len(b) - pocet rvnic
            L = 0
            for j in range(len(b)): # len(b) - zde se rovna poctu neznamych
                if j != i:
                    L += A[i][j]*x[j]
            x_next[i] = (b[i]-L)/A[i][i]
        deviation[l] = frob_norm(np.subtract(x_next,x))
        np.copyto(x, x_next)
        l += 1
    return x, deviation


def superrelaxacni_m(A,b,iter,omega):
    x = np.zeros(len(b))
    np.copyto(x,b)
    x_next = np.zeros(len(b))
    np.copyto(x_next,b)
    deviation = np.zeros(iter)
    l = 0
    while l < iter:
        for i in range(len(b)):
            L = 0
            for j in range(len(b)):
                if j != i:
                    L += A[i][j]*x_next[j]
            x_next[i] = (1 - omega)*x[i] + (omega/A[i][i])*(b[i]-L)
        deviation[l] = frob_norm(np.subtract(x, x_next))
        np.copyto(x, x_next)
        l += 1
    return x, deviation


# main function
def main():

    poradi = 15.0 # jachym smid - 15
    
    pocet_iteraci = 10 
    p1 = 0.4*(np.exp(1/poradi))/(poradi+9)
    p2 = poradi/5.0
    A1 = np.array([[4.0, -1.0, -6.0],[-5.0, -4.0, 10.0],[0.0, 9.0, 4.0]])
    A2 = np.array([[1.8, -0.1, p1],[-1.0, 0.2, 0.0],[-1.0, 1.0, 0.6]])
    b = np.array([p2, 2.0, 0.5])
    omega = 1.1


    # vypocet
    res_pim, dev_pim = prosta_m(A2, b, pocet_iteraci)
    res_gauss, dev_gauss = gauss_seidel_m(A2, b, pocet_iteraci)
    res_jacobi, dev_jacobi = jacobiho_m(A2, b, pocet_iteraci)
    res_sor, dev_sor = superrelaxacni_m(A2, b, pocet_iteraci, omega)


    print(f"Výsledek PIM: {res_pim}\nVýsledek JM: {res_jacobi}\nVýsledek GSM: {res_gauss}\nVýsledky SOR: {res_sor}\n")
    print(f"Odchylky PIM: {dev_pim}\nOdchylky JM: {dev_jacobi}\nOdchylky GSM: {dev_gauss}\nOdchylky SOR: {dev_sor}\n")
    
    
    x_axis = np.linspace(0,pocet_iteraci-1,pocet_iteraci)
    fig, ax = plt.subplots(nrows=2, ncols=2, layout = 'tight')
    fig.suptitle("Rychlost konvergence jednotlivých metod")

    ax[0, 0].plot(x_axis,dev_pim, color = 'red', label = "PIM")
    ax[0, 0].set_title("Konvergence PIM")
    ax[0, 0].set_xlabel("Počet iterací")
    ax[0, 0].set_ylabel("Error")
    ax[0, 0].grid()

    ax[0, 1].plot(x_axis,dev_jacobi, color = 'blue', label = "JM")
    ax[0, 1].set_title("Konvergence JM")
    ax[0, 1].set_xlabel("Počet iterací")
    ax[0, 1].set_ylabel("Error")
    ax[0, 1].grid()

    ax[1, 0].plot(x_axis, dev_gauss, color = 'green', label = "GSM")
    ax[1, 0].set_title("Konvergence GSM")
    ax[1, 0].set_xlabel("Počet iterací")
    ax[1, 0].set_ylabel("Error")
    ax[1, 0].grid()

    ax[1, 1].plot(x_axis, dev_sor, color = 'black', label = "SOR")
    ax[1,1].set_title("Konvergence SOR")
    ax[1,1].set_xlabel("Počet iterací")
    ax[1,1].set_ylabel("Error")
    ax[1,1].grid()


    #citlivot SOR na omega
    plt.figure(2)
    plt.suptitle("Citlivost SOR na $\omega$")
    omeg = np.linspace(0,2,11)
    for i in range(len(omeg)):
        if i == 0: continue
        ax2 = plt.subplot(2,5,i)
        y, z = superrelaxacni_m(A2, b, pocet_iteraci, omeg[i])
        ax2.plot(x_axis,z)
        ax2.set_title(f"$\omega$ = {omeg[i]}")
        ax2.set_xlabel("Počet iterací")
        ax2.set_ylabel("Error")
        ax2.grid()
    plt.show()



if __name__ == "__main__":
    main()
