Colab exploration: https://colab.research.google.com/drive/1JG_pUasVvcPLgLLpns_hRYk69Ha-CwMw?authuser=1#scrollTo=f5-KPRwXySbE
# VAE
## Variational encoder
$$
\begin{align*}
e_\phi(x) &= \left(\mu_\phi(x), \log \sigma^2_\phi(x)\right) \\
q_\phi(z \mid x) &= \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2 (x))) \\
z &\sim q_\phi(z \mid x)
\end{align*}
$$
