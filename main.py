import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpf
from statistics import mean
from math import *
import warnings
warnings.simplefilter('error', category=RuntimeWarning)  # RuntimeWarningを例外扱いに設定


def logistic_map(a:float, x:'numpy.ndarray'):
    """Logistic Map

    Args:
        a (float): パラメータ.
        x (numpy.ndarray): 軌道. x_k

    Returns:
        numpy.ndarray: x_k+1
    """
    return a * x * (1 - x)


def logistic_map_orbit(seed:float, a:float, n_iter:int, n_skip:int=0):
    """Logistic Mapの軌道を描写

    Args:
        seed        (float): x_0の値
        a           (float): Logistic Mapのパラーメータ
        n_iter        (int): 反復回数
        n_skip (int, optional): 反復を無視する回数. Defaults to 0.
    """
    X_t = []
    T = []
    t = 0
    x = seed
    # 反復計算
    for i in range(n_iter + n_skip):
        if i >= n_skip:
            X_t.append(x)
            T.append(t)
            t+=1
        x = logistic_map(a=a, x=x)
    # Stepと軌道Xをplot
    plt.plot(T, X_t)
    plt.ylim(0, 1)
    plt.xlim(0, T[-1])
    plt.xlabel('Step t')
    plt.ylabel('X_t')
    plt.savefig(f"logistic_map_orbit_seed{seed}_a{a}_iter{n_iter}_skip_{n_skip}.png")
    plt.clf()
    plt.close()


def bifurcation_diagram_for_logistic_map(seed:float, n_skip:int, n_iter:int, step:float=0.0001, a_min:float=0):
    """Logistic Mapの分岐図を作成し、Lyapunov exponentsを計算する

    Args:
        seed (float): x_0
        n_skip (int): 反復を無視する回数.
        n_iter (int): 反復回数.
        step (float, optional): パラメータaを設定するstep数. Defaults to 0.0001.
        a_min (float, optional): パラメータaの最小値. Defaults to 0.
    
    Returns:
        mean_lyapunov (float): 平均リアプノフ指数
    """
    A = []
    X = []
    lyapunov = []
    
    a_range = np.linspace(a_min, 4, int(1/step))

    for a in a_range:
        x = seed
        for i in range(n_iter + n_skip + 1):
            if i >= n_skip:
                A.append(a)
                X.append(x)
                # lyapunov.append(log(abs(1.0 - 2.0 * x)))
                lyapunov.append(log(abs(a - 2.0 * a * x)+1e-7))
            x = logistic_map(a=a, x=x)
    # Plot the 分岐図  
    plt.plot(A, X, ls='', marker=',', c="black")
    plt.ylim(0, 1)
    plt.xlim(a_min, 4)
    plt.xlabel('a')
    plt.ylabel('X')
    plt.savefig(f'logistic_map_bifurcation_seed{seed}_skip{n_skip}_iter{n_iter}_step{step}_amin_{a_min}.png')
    plt.clf()
    plt.close()
    # Plot the Lyapunov exponents diagram
    plt.plot(A, lyapunov, ls='', marker=',', c="blue")
    plt.ylim(-7, 1)
    plt.xlim(a_min, 4)
    plt.xlabel('a')
    plt.ylabel('lyapunov exponent')
    plt.savefig(f'logistic_map_lyapunov_seed{seed}_skip{n_skip}_iter{n_iter}_step{step}_amin_{a_min}.png')
    plt.clf()
    plt.close()
    # plot 分岐図 and the Lyapunov exponents diagram
    plt.plot(A, X, ls='', marker=',', c="black")
    plt.plot(A, lyapunov, ls='', marker=',', c="blue")
    plt.ylim(-5, 1)
    plt.xlim(a_min, 4)
    plt.xlabel('a')
    plt.ylabel('X, lyapunov exponent')
    plt.savefig(f'logistic_map_bifurcation_and_lyapunov_seed{seed}_skip{n_skip}_iter{n_iter}_step{step}_amin_{a_min}.png')
    plt.clf()
    plt.close()
    
    return mean(lyapunov)


def henon_map(x:float, y:float, a:float, b:float):
    """Henon Map

    Args:
        x (float): x_k
        y (float): y_k
        a (float): パラメータa
        b (float): パラメータb

    Returns:
        float: x_k+1, y_k+1
    """
    try:
        return 1.0 - a * x * x + y, b * x
    except Exception as e:
        q = a * x * x
        w = b * x
        z = 1.0 - q
        return 1.0 - a * x * x + y, b * x


def bifurcation_diagram_for_henon_map(x_0:float, y_0:float, b:float, n_skip:int, n_iter:int, step:float=0.0001, a_min:float=0):
    """Henon Mapの分岐図を作成する

    Args:
        x_0 (float): x_0
        y_0 (float): y_0
        b   (float): パラメータb
        n_skip (int): 反復を無視する回数.
        n_iter (int): 反復回数.
        step (float, optional): パラメータaを設定するstep数. Defaults to 0.0001.
        a_min (float, optional): パラメータaの最小値. Defaults to 0.
    """
    delta = 1e-4
    
    A = []
    X = []
    
    # for Lyapunov exponents
    lyapunov = []
    e_0 = np.array([[-1.0 / sqrt(2.0)], [1.0 / sqrt(2.0)]])
    f_0 = np.array([[1.0 / sqrt(2.0)], [1.0 / sqrt(2.0)]])
    
    a_range = np.linspace(a_min, 1.41, int(1/step))

    for a in a_range:
        x = x_0
        y = y_0
        for i in range(n_iter + n_skip + 1):
            if i >= n_skip:
                A.append(a)
                X.append(x)
                lyapunov.append(log(l))
            # ヤコビ行列を作成
            DF = np.array([[-2.0*a*x, 1.0], [b, 0.0]])
            # e_n+1, f_n+1を計算
            e_1 = np.dot(DF, e_0)
            f_1 = np.dot(DF, f_0)
            # line elementを計算
            l = sqrt(e_1[0][0]**2.0 + e_1[1][0]**2.0)
            # e_n+1に直行するベクトルf'_n+1を計算
            work = sum(f_1 * e_1) / (l * l + delta)
            f_dash = f_1 - np.dot(work[0], e_1)
            f_dash_norm = sqrt(f_dash[0][0]**2.0 + f_dash[1][0]**2.0)
            # e_0, f_0を正規化
            e_0 = e_0 / (l + delta)
            f_0 = f_0 / (f_dash_norm + delta)
            # henon map
            x, y = henon_map(a=a, b=b, x=x, y=y)
            
    # Plot 分岐図
    plt.plot(A, X, ls='', marker=',', c="black")
    plt.ylim(-1.5, 1.5)
    plt.xlim(1, 1.5)
    plt.xlabel('a')
    plt.ylabel('X')
    plt.savefig(f'henon_map_bifurcation_x{x_0}_y_{y_0}_b{b}_skip{n_skip}_iter{n_iter}_step{step}_amin_{a_min}.png')
    plt.clf()
    plt.close()
    # Plot the Lyapunov exponents diagram
    plt.plot(A, lyapunov, ls='', marker=',', c="blue")
    plt.ylim(-3, 3)
    plt.xlim(1, 1.5)
    plt.xlabel('a')
    plt.ylabel('lyapunov')
    plt.savefig(f'henon_map_lyapunov_x{x_0}_y_{y_0}_b{b}_skip{n_skip}_iter{n_iter}_step{step}_amin_{a_min}.png')
    plt.clf()
    plt.close()
    # plot 分岐図 and the Lyapunov exponents diagram
    plt.plot(A, X, ls='', marker=',', c="black")
    plt.plot(A, lyapunov, ls='', marker=',', c="blue")
    plt.ylim(-1, 2)
    plt.xlim(1, 1.5)
    plt.xlabel('a')
    plt.ylabel('X, lyapunov')
    plt.savefig(f'henon_map_bifurcation_and_lyapunov_x{x_0}_y_{y_0}_b{b}_skip{n_skip}_iter{n_iter}_step{step}_amin_{a_min}.png')
    plt.clf()
    plt.close()
    
    return sum(lyapunov) / len(lyapunov)    


if __name__ == "__main__":
    # Logistic Mapを可視化
    x = np.linspace(0, 1)
    plt.plot(x, logistic_map(a=2, x=x), 'k')
    plt.savefig('logistic_map.png')
    plt.clf()
    plt.close()
    
    # Logistic Mapの軌道を可視化
    logistic_map_orbit(seed=0.1, a=3.05, n_iter=100)
    logistic_map_orbit(seed=0.1, a=3.9, n_iter=100)
    logistic_map_orbit(seed=0.1, a=3.9, n_iter=100, n_skip=1000)
    
    # Logistic Mapの分岐図とリアプノフ指数の図を作成
    print(f"Average of Lyapnov expontns (Logistic map) = {bifurcation_diagram_for_logistic_map(seed=0.2, n_skip=100, n_iter=5)}")
    print(f"Average of Lyapnov expontns (Logistic map) = {bifurcation_diagram_for_logistic_map(seed=0.2, n_skip=100, n_iter=10)}")
    print(f"Average of Lyapnov expontns (Logistic map) = {bifurcation_diagram_for_logistic_map(seed=0.2, n_skip=100, n_iter=10, a_min=2.8)}")
    
    # Henon Mapを可視化
    x, y, a, b = mpf(-.75), mpf(.32), mpf(1.4), mpf(0.3)
    mp.dps = 10
    xt, yt = [], []
    for _ in range(10000 + 1):
        xn, yn = henon_map(x, y, a, b)
        xt.append(x)
        yt.append(y)
        x, y = xn, yn
    plt.scatter(xt, yt, c="blue", s=0.1)
    plt.xlabel(r"$X_n$")
    plt.ylabel(r"$Y_n$")
    plt.savefig(f'henon_map_x{x}_y{y}_a{a}_b{b}.png')
    plt.clf()
    plt.close()
    
    # Henon Mapの分岐図を作成
    mean_lyapunov_henon_map = bifurcation_diagram_for_henon_map(
        x_0=-0.75, y_0=0.32, b=0.3, step=0.001,n_iter=150, n_skip=100, a_min=0
    )
    print(f"Average of Lyapnov expontns (Henon map) = {mean_lyapunov_henon_map}")
    mean_lyapunov_henon_map = bifurcation_diagram_for_henon_map(
        x_0=-0.75, y_0=0.32, b=0.3, step=0.001,n_iter=150, n_skip=100, a_min=1.01
    )
    print(f"Average of Lyapnov expontns (Henon map) = {mean_lyapunov_henon_map}")
