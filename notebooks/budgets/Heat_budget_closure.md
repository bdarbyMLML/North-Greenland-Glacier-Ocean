# Heat Budget Closure for MITgcm

### Heat budget equation
The heat budget for the MITgcm is based on this equation:

$\frac{\partial \theta}{\partial t}$ $=$ $-$ $\nabla$ $\cdot$ $(\theta$ $\mathbf{u})$ $-$ $\nabla$ $\cdot$ $F_{\textrm{diff}}^{\theta}$ $+$ ${F}_\textrm{forc}^{\theta}$

where, $\frac{\partial \theta}{\partial t}$ is the change in potential temperature over time. The equation above describes this change as a function of the convergence (the $-\nabla \cdot$ operator) of heat advection and diffusion with the addition of any forcing terms.

In order to simulate this equation and be used for MITgcm, a two steps are needed. 1) A coordinate change to best represent the free surface with fixed depth bins $z^* = \frac{z - \eta}{H + \eta}H $, where $\eta$ and $H$ are the displacement of the ocean surface and the ocean depth respectively, and 2) the residual mean velocities,

$v_{res}$ = $(u_{res},v_{res},w_{res})$ $=$ $(u,v,w)$ $+$ $(u_b,v_b,w_b)$, 

that parameterizes unresolved eddies. From step 1, our coordinate change introduces a scaling factor $s^* = 1+ \frac{\eta}{H}$ as well as a change to our horizontal and vertical divergences, $` ( \nabla_z^* `$ and $\frac{\partial}{\partial z^{*}} )$, which are seperated, leading to our final equation:

$`\frac{\partial(s^*\theta)}{\partial t}$ $=$ $-\nabla_{z^{*}}$ $\cdot(s^*\theta\,\mathbf{v}_{res}- \frac{\partial(\theta\,w_{res})}{\partial z^{*}}- s^* ({\nabla\cdot\mathbf{F}_\textrm{diff}^{\theta}+s^* {F}_\textrm{forc}^{\theta}`$

In order to make the calculation a bit easier on the eyes the different budget terms will be seperated into: 

$`\underbrace{\frac{\partial(s^*\theta)}{\partial t}}_{G^{\theta}_\textrm{total}} = \underbrace{-\nabla_{z^{*}} \cdot(s^*\theta\,\mathbf{v}_{res}) - \frac{\partial(\theta\,w_{res})}{\partial z^{*}}}_{G^{\theta}_\textrm{advection}}\underbrace{- s^* ({\nabla\cdot\mathbf{F}_\textrm{diff}^{\theta}})}_{G^{\theta}_\textrm{diffusion}} + \underbrace{s^* {F}_\textrm{forc}^{\theta}}_{G^{\theta}_\textrm{forcing}}`$

Making our budget equation: $`G^{\theta}_\textrm{total} = G^{\theta}_\textrm{advection} + G^{\theta}_\textrm{diffusion} + G^{\theta}_\textrm{forcing}`$

### Converting to MITgcm output

#### Calculating $G^{\theta}_\textrm{total}$
In order to calculate the $G^{\theta}_\textrm{total}$ snapshots of our model feild are needed for $\eta$ and $\theta$. This is because averaging these fields would smooth out variability and we would lose information.

$`G^{\theta}_\textrm{total} = \frac{\partial(s^*\theta)}{\partial t} = \frac{\partial(1+ \frac{\eta}{H}) \theta}{\partial t}`$
in terms of our variables our $G^{\theta}_\textrm{total}$ would be:
$=$ 
```sTHETA.diff(dim = 'time') = THETA_snapshot*(1+ETAN_snapshot/grid.Depth).diff(dim = 'time')```

where ```.diff(dim= 'time')``` is the numpy difference function with respect to the time column, and grid.Depth is the Depth vaiable from out MITgcm grid.

### Calculating $G^{\theta}_\textrm{advection}$

At first glance, in order to calculate our $`G^{\theta}_\textrm{horizontal advection} = -\nabla_{z^{*}} \cdot(s^*\theta\,\mathbf{v}_{res})`$  equation, some tough coding are needed. Fortunately, there exists output of the $\theta$ advection terms in the x and y.

Making our equation $G^{\theta}_\textrm{advection}$:
$`=`$
```-1*(ADVx_TH.diff(dim='X') + ADVy_TH.diff(dim='Y'))```


next we calculate our vertical advection term $G^{\theta}_\textrm{vertical advection} = - \frac{\partial(\theta\,w_{res})}{\partial z^{*}}$

$=$
```ADVr_TH.diff(dim='Z')```

putting it all together we have,

$G^{\theta}_\textrm{advection}$ = ```(-1*(ADVx_TH.diff(dim='X') + ADVy_TH.diff(dim='Y')) + ADVr_TH.diff(dim='Z'))/volume```

where,
```volume = (grid.rA*grid.drF*grid.hFacC)```



#### Calculating $G^{\theta}_\textrm{diffusion}$


The form of our diffusion term is almost identical to our advection term except for the variables used. Our horizontal and vertical components are:
$G^{\theta}_\textrm{horizontal diffusion}$
$=$
```-1*(DFxE.diff(dim='X') + DFyE.diff(dim='Y')))```

and,

$G^{\theta}_\textrm{vertical diffusion}$

$=$
```DFrE_TH.diff(dim='Z') + DFrI_TH.diff(dim='Z')```
Making our final $G^{\theta}_\textrm{diffusion}$ term:

$G^{\theta}_\textrm{diffusion} = $ ``` (-1*(DFxE.diff(dim='X') + DFyE.diff(dim='Y')) + DFrE_TH.diff(dim='Z') + DFrI_TH.diff(dim='Z'))/volume```

If you are using the KPP parameter, the KPP term has to be added to the $G^{\theta}_\textrm{diffusion}$.

$G^{\theta}_\textrm{diffusion with kpp} = $ ``` (-1*(DFxE.diff(dim='X') + DFyE.diff(dim='Y')) + DFrE_TH.diff(dim='Z') + DFrI_TH.diff(dim='Z')+KPPg_TH.diff(dim='Z'))/volume```

#### Calculating $G^{\theta}_\textrm{forcing}$

To complete our heat budget we must finally calculate our forcing term. 






```# Seawater density (kg/m^3)rhoconst = 1029
## needed to convert surface mass fluxes to volume fluxes

# Heat capacity (J/kg/K)
c_p = 3994

# Constants for surface heat penetration (from Table 2 of Paulson and Simpson, 1977)
R = 0.62
zeta1 = 0.6
zeta2 = 20.0 
```




