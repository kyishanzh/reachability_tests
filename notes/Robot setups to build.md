- Rotary link without joint limits
- Rotary link with joint limits


## Excavator 3D robot setup?

![[types of joints.png]]
(Rotary = Revolute)
- Rotary joints allow 1DoF (rotation around fixed axis)

## Rotary link robot setup
### 2-link planar arm on base $(x, y)$ with no joint limits
#### Setup
- Robot configuration $Q = (x, y, \theta_1, \theta_2) \in \mathbb{R}^4$ [4 DoF]
	- Leaving out base heading $\psi$ from free variables for now (assuming robot must be facing target → fixed orientation to $H$ once we decide base $(x, y)$)
- Target $H = (h_x, h_y) \in \mathbb{R}^2$
- **Goal:** Learn the distribution of $p(Q \mid H)$
#### Forward kinematics $f: Q \rightarrow H$
$$x_e = b+R(\psi) (L_1 u(\theta_1) + L_2u(\theta_1 + \theta_2)), \quad u(\alpha) = (\cos \alpha, \sin \alpha)$$
(Later can add base heading $\psi$, or enforce that base must be facing target in which case it is determined separately [or allow for noise range])
#### (IK) Sample Q given H $f^{-1} : H \rightarrow Q$


#### What should true distribution $p(Q \mid H)$ look like?
- Some notes:
	- Gauge freedom between $\psi$ and $\theta_1$: only the sum (world orientation of link 1) matters: $\alpha = \psi + \theta_1$
		- So shifting $\psi \rightarrow \psi + \delta, \, \theta_1 \rightarrow \theta_1 - \delta$ does not change the world geometry of the arm [this might make training tough since there are many mathematically "different" but physically identical configurations]
		- [!] **Break this gauge freedom** by enforcing that the robot should be facing the target $H$ (restricts $\psi$, can allow some noise in defining "face the target")
		- Later: add more realistic constraints to break gauge freedom (e.g. robot camera should be facing the task/task should be visible to robot)
	- Valid $Q$'s form donut ring around $H$
#### What are the modes present (to evaluate mode collapse)?
- 2 discrete branches: elbow up/down for rotary link
- Should have variance in base heading (or else robot is learning some fixed thing)
#### How to generate data
- Problem with $Q \rightarrow H$ sampling: 
	- If we sample $Q$ uniformly, the resulting $H$ positions will follow a non-uniform density across the workspace → model will learn this bias and perform worse at workspace spots that got sparse coverge.
- Instead: Control the distribution of $H$ to be uniform across the task space → sample $H$ first, then sample the valid $Q$s that achieve it.
1. **Sample the target** $H = (h_x, h_y)$ from our desired task distribution, i.e. $$H \sim \text{Uniform}(\text{workspace bounds})$$
2. **Sample base position** $(x,y)$ via the "donut":
	- For the arm to reach $H$, the base must be at a specific distance $r$ away from $H$:
		- Inner radius $R_{\text{min}}$: $|L_1 - L_2|$ (folded in)
		- Outer radius $R_{\text{max}}$: $L_1 + L_2$ (fully extended)
	- Sample the base position in polar coordinates centered at $H$:
		1. Sample distance $r$ → sample $r^2$ uniformly or sample $r$ linearly, e.g. potentially can do:$$r \sim \text{Uniform}(R_{\min} + \epsilon , R_{\max} - \epsilon)$$ (^ $\epsilon$ proxy for high manipulability → probably want to avoid $R_\min$ and $R_\max$ exactly in the real world to avoid singularities)
		2. Sample direction $\phi$ (where the base is around the donut): $$\phi \sim \text{Uniform}(0, 2\pi)$$
		3. Compute base coordinates:$$\begin{align*} x &= h_x + r\cos (\phi) \\ y &= h_y + r \sin (\phi)\end{align*}$$
3. **Handle orientation** $\psi$ (breaking gauge freedom): ==general constraint = robot faces target!==
	1. Ideal heading: $\psi_{\text{ideal}} = \text{atan2} (h_y - y, h_x - x)$ 
	2. Add noise: To prevent the model from collapsing to a single deterministic mapping (and to allow the network to learn correction), add noise: $$\psi = \psi_{\text{ideal}} + \delta, \quad \delta \sim \mathcal{N}(0, \sigma^2)$$
	- [!] **Beware:** if $\delta$ is too large, the target may fall out of the arm's workspace if the arm has joint limits (do some plotting to verify that data generally looks ok)
4. **Analytical arm IK** (solve for $\theta_1, \theta_2$):
	- Now that base $B = (x, y, \psi)$ is fixed, transform $H$ into the robot's local frame.
	1. Local target: $$h_{\text{local}} = R(\psi)^\top (H - B[0:2]) := (u,v)$$
	2. Solve $\theta_2$ (elbow): Using law of cosines: $$\cos(\theta_2) = \frac{u^2 + v^2 - L_1^2 - L_2^2}{2L_1L_2}$$
		- If $|\cos(\theta_2)|>1$, the base was sampled too far/close (shouldn't happen if Step 2 is correct) [read into the math of this]
		- [!] **Important:** There are two solutions $\theta_2^+ = \arccos(\ldots), \theta_2^- = \arccos(\ldots)$ . **Save both** as separate data points $(Q_1, H)$ and $(Q_2, H)$. This explicitly teaches the multimodal distribution (elbow up vs. down)
	3. Solve $\theta_1$ (shoulder): $$\theta_1 = \text{atan2}(u, v) - \text{atan2}(L_2 \sin \theta_2, L_1 + L_2 \cos \theta_2)$$
5. **Filter**: Check if $\theta_1$ and $\theta_2$ are within joint limits. If yes, add to dataset.
6. (If we want to address real world priors in the future by biasing the dataset -- e.g. adding cost functions for manipulability, clearance) → to incorporate these into the dataset generation:
	- Rejection sampling filter: After generating a candidate $Q$ using steps above, calculate a score $S(Q)$: $$S(Q) = w_1 \cdot \text{Manipulability}(Q) + w_2 \cdot \text{LimitMargin}(Q)$$
		- Hard filter: Discard if $S(Q) < \text{threshold}$
		- Soft filter (probabilistic): Accept $Q$ with probability $P = \frac{S(Q)}{S_\max}$
		- This creates a dataset where "better" configurations are more dense → the generative model will learn to output high-quality configurations more often because they appear more frequently in the training distribution
7. **Visualizing the data distribution:** (to validate our data and later the model)
	- Generate a validation set where $H$ is fixed at $(0, 0)$
	1. Plot 1: The base donut (workspace density)
		- Scatterplot of base $(x, y, \psi)$ for fixed $H = (0, 0)$
		- Expectation: Should see a ring + orientation arrows for this points should all point inward toward the origin
	2. Plot 2: The joint modes
		- Scatter plot of $\theta_1$ vs $\theta_2$
		- Should see two distinct clusters representing the elbow-up and elbow-down solutions. If the model interpolates between these clusters (e.g. draws samples in the middle), it is suffering from mode collapse/averaging.
[Visualize robot as a tiny rectangle with a circle on one of the sides maybe instead of point? To be able to visualize the base heading -- or have a point with an orientation arrow]

---
1. Sample targets $H$ from the task distribution (assume uniform across workspace?)
2. [Later] Sample environment context if applicable (obstacles, regions)
3. Enumerate candidate bases inside workspace
4. For each candidate $B$ (and $\psi$ if applicable), run IK and collect all valid branches (elbow up/down) *that satisfy joint limits.*
5. [Later] Score each candidate solution with a cost that mirrors real-world preferences:
	1. Joint-limit margin
	2. Smoothness from current posture
	3. Manipulability/avoid singularity
	4. Collision clearance
	5. Base motion penalty
	- etc.!
#### Implementation notes
- Would be nice to have a plot of true data distribution $p(Q \mid H)$ that we can compare with learned $p^* (Q \mid H)$ [maybe a heat map of some sort?]

### 2-link planar arm on base $(x, y, \psi)$ with joint limits $[-\pi, \pi]$




---
## Archived notes
2-link planar arm on a translating base
### Setup
![[rotary link env  pic drawing.png|300]]
- Desired hand position $H = (h_x, h_y)$
- Robot config $Q = (x, y, \theta_1, \theta_2)$ with link lengths $L_1, L_2$ where $\theta_1$ is the orientation of link 1 in the world frame, $\theta_2$ is the rotation of link 2 relative to link 1
	- Absolute orientation of link 2 is $\theta_1 + \theta_2$
	- Angle limits: enforce $\theta_1 \in (-\pi, \pi]$ and $\theta_2 \in (-\pi, \pi]$)
	- [*] Add a base $\psi$ as next-step advancement
### FK
- Base: $b = (b_x, b_y)$
- Endpoint of link 1: $e_1 = b + L_1(\cos \theta_1, \sin\theta_1)$
- Endpoint of link 2 (end-effector position): $e_2 = e_1 + L_2(\cos (\theta_1, \theta_2), \sin(\theta_1 + \theta_2))$
### Multimodality
- Elbow-up/elbow-down ambiguity: $\theta_2 = \pm \arccos(\cdot)$ → $p(Q \mid H)$ will have two separated modes in $\theta_2$
![[rotary link env modes.png]]
More math:

### Next step extensions from this environment
- Base $(x,y) \rightarrow (x, y, \psi)$ 
	- $e_2 = b + R(\psi) (L_2 u(\theta_1) + L_2 u(\theta_1 + \theta_2))$