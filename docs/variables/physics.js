// Definitions explain the physical model, with code mappings kept as a separate layer.
export const normalizeTex = tex => String(tex || '').replace(/\\\\(?=[A-Za-z])/g, '\\');

export const CONCEPTS = [
  {id:'heater', name:'heater power', symbol:'P_H', unit:'W', position:[0,0],
    question:'How does electrical drive heat the ring?',
    meaning:'the heater turns electrical power into heat. That heat reaches the ring through the thermal network.',
    equation:String.raw`P_H=V_H I_H=I_H^2R_H=\frac{V_H^2}{R_H}`,
    note:'these equalities describe the same resistor. The drive law determines which voltage, current, or power is held fixed.',
    definitions:[['P_H','heater power','W'],['V_H','voltage across the heater','V'],['I_H','heater current','A'],['R_H','heater resistance at its current temperature','\\Omega']],
    records:['ring_float.he_p'], source:'mrm/plant/electrical.py · heater_power_row'},
  {id:'temperature', name:'ring temperature', symbol:String.raw`T_{\mathrm{ring}}`, unit:'K', position:[1,0],
    question:'How do heat sources set the ring temperature?',
    meaning:'each thermal path stores a temperature rise and relaxes over its own time constant. The ring temperature adds the paths that reach the ring.',
    equation:String.raw`\tau_j\dot x_j=-x_j+r_jP_{\mathrm{src}(j)}`,
    extra:String.raw`T_{\mathrm{ring}}=T_{\mathrm{base}}+\sum_{j\to\mathrm{ring}}x_j`,
    note:'the model has several thermal states. Absorbed optical power and heater power enter through distinct heat paths.',
    definitions:[['T_{\\mathrm{ring}}','absolute ring temperature','K'],['\\dot x_j','rate of change of the thermal temperature rise','K/s'],['j','thermal path index','1'],['x_j','temperature rise stored in thermal path j','K'],['\\tau_j','thermal relaxation time of that path','s'],['r_j','temperature rise per source watt at steady state','K/W'],['P_{\\mathrm{src}(j)}','power driving that path','W'],['T_{\\mathrm{base}}','base temperature','K']],
    records:['ring_float.t_ring','ring_float.t_new','S.th_x','S.th_t_obs'], source:'mrm/plant/thermal.py · foh_scalar and thermal_outputs_b'},
  {id:'resonance', name:'resonance wavelength', symbol:String.raw`\lambda_r`, unit:'m', position:[2,0],
    question:'How does temperature move the resonance?',
    meaning:'a warmer ring changes its refractive index. The resonance moves by the integral of its thermal tuning slope.',
    equation:String.raw`\Delta\lambda_r=\int_{T_{\mathrm{ref}}}^{T_{\mathrm{ring}}}s(T)\,\mathrm dT`,
    extra:String.raw`s(T)=s_0[1+\beta_T(T-T_0)]`,
    extra2:String.raw`\lambda_r=\lambda_{r,\mathrm{ref}}+\Delta\lambda_r`,
    note:'bias is held fixed in this diagram. The full model also adds the junction-induced resonance shift. Detuning uses resonance wavelength minus laser wavelength.',
    definitions:[['T_{\\mathrm{ring}}','current ring temperature','K'],['\\lambda_{r,\\mathrm{ref}}','resonance at the reference temperature and held bias','m'],['\\lambda_r','current resonance wavelength','m'],['\\Delta\\lambda_r','shift from the reference resonance','m'],['s_0','thermal tuning slope at T₀','m/K'],['\\beta_T','relative change of tuning slope per kelvin','1/K'],['T_0','temperature where the slope equals s₀','K'],['T_{\\mathrm{ref}}','reference ring temperature','K'],['\\lambda_L','fixed laser wavelength','m'],['\\lambda_r-\\lambda_L','detuning, positive when the resonance is red of the laser','m']],
    records:['ring_float.det','pp.optics.lambda_res_cold_m','cfg.ring.thermal.dlambda_dt_pm_per_k'], source:'mrm/plant/phase.py · thermal_shift_m and resonance_shift_row'},
  {id:'absorption', name:'absorbed optical power', symbol:String.raw`P_{\mathrm{abs}}`, unit:'W', position:[0,1],
    question:'How does light feed back into temperature?',
    meaning:'only the absorbed fraction of intrinsic optical loss becomes heat. Temperature changes that absorption by shifting the optical response.',
    equation:String.raw`P_{\mathrm{abs}}=\eta_{\mathrm{heat}}P_{\mathrm{loss}}`,
    note:'follow the return arrow to temperature. A temperature increase can raise or lower absorption, depending on the operating point. This is the feedback behind thermal self-stabilization and bistability.',
    definitions:[['P_{\\mathrm{abs}}','optical power converted to heat','W'],['\\eta_{\\mathrm{heat}}','absorbed fraction of intrinsic optical loss','1'],['P_{\\mathrm{loss}}','total intrinsic optical loss power','W']],
    records:['ring_float.p_abs','selfheat.r_abs_sum','selfheat.e0'], source:'mrm/plant/optics.py · absorbed_power_at_trial_temperature; mrm/plant/selfheat.py · solve_ring_temperature'},
  {id:'loss', name:'intrinsic optical loss', symbol:String.raw`P_{\mathrm{loss}}`, unit:'W', position:[1,1],
    question:'Where does the incoming optical power go?',
    meaning:'incoming power leaves through the optical ports or is lost inside the ring. Intrinsic loss includes absorption, scattering, and radiation.',
    equation:String.raw`P_{\mathrm{loss}}=P_{\mathrm{in}}-\sum_{p\in\mathrm{ports}}P_{p,\mathrm{out}}`,
    note:'this power balance applies to the exact ring model. Intrinsic loss is not all heat. The absorption fraction selects the part that heats the device.',
    definitions:[['p','output port index','1'],['P_{\\mathrm{in}}','total incident optical power','W'],['P_{p,\\mathrm{out}}','power leaving one output port','W'],['P_{\\mathrm{loss}}','intrinsic loss across both travelling directions','W']],
    records:['pp.optics.alpha_i_per_m'], source:'mrm/plant/optics.py · RingResponse.p_loss and ring_exact'},
  {id:'phase', name:'round-trip phase', symbol:String.raw`\Phi`, unit:'rad', position:[2,1],
    question:'Why does shifting resonance change the optical response?',
    meaning:'light interferes with itself after each ring trip. The phase controls whether successive trips reinforce each other at the laser wavelength.',
    equation:String.raw`\Phi=\frac{2\pi n_{\mathrm{eff}}(x)L}{x}+\psi`,
    extra:String.raw`x=\lambda_L-\Delta\lambda_{\mathrm{total}}`,
    extra2:String.raw`\Phi_{\mathrm{res}}=2\pi m`,
    note:'the resonance condition makes successive round trips add in phase. the code shifts the cold phase curve and evaluates it at the fixed laser. the additional coupler phase is usually zero when the measured cold resonance already includes it.',
    definitions:[['\\lambda_L','fixed laser wavelength','m'],['\\Delta\\lambda_{\\mathrm{total}}','total thermal and junction shift from the cold reference','m'],['\\Phi','total phase including the coupler phase','rad'],['\\Phi_{\\mathrm{res}}','phase at resonance','rad'],['n_{\\mathrm{eff}}(x)','effective index on the cold phase curve','1'],['L','ring round-trip length','m'],['x','wavelength coordinate on the shifted cold curve','m'],['\\psi','additional coupler phase','rad'],['m','integer resonance order','1']],
    records:['selfheat.phi0r','selfheat.phi1r','P.ph_ring'], source:'mrm/plant/phase.py · _fill_level; mrm/plant/optics.py · ring_exact'},
  {id:'current', name:'photodiode current', symbol:String.raw`I_{\mathrm{PD}}`, unit:'A', position:[1,2],
    question:'How does drop-port light become a measured signal?',
    meaning:'the photodiode converts drop power into photocurrent. Dark current adds a background that rises with photodiode temperature.',
    equation:String.raw`\overline I_{\mathrm{PD}}=\mathcal R P_{\mathrm{drop}}+I_{\mathrm{dark}}`,
    extra:String.raw`I_{\mathrm{dark}}=I_0\exp\!\left[-\frac{E_a}{k_B}\left(\frac1{T_{\mathrm{PD}}}-\frac1{T_{\mathrm{PD},0}}\right)\right]`,
    note:'the first equation is the mean signal before extra background pickup, gain errors, and noise. Integration, ADC conversion, delivery delays, and controller codes belong to the model & simulation view.',
    definitions:[['\\overline I_{\\mathrm{PD}}','mean photodiode current','A'],['P_{\\mathrm{drop}}','optical power reaching the photodiode','W'],['\\mathcal R','photodiode responsivity','A/W'],['I_{\\mathrm{dark}}','current without incident light','A'],['I_0','dark current at the reference temperature','A'],['E_a','activation energy in joules','J'],['k_B','Boltzmann constant','J/K'],['T_{\\mathrm{PD}}','photodiode temperature','K'],['T_{\\mathrm{PD},0}','reference photodiode temperature','K']],
    records:['ring_float.i_dark','ring_float.t_pd','obs.tx.monitor.latest'], source:'mrm/plant/sensing.py · dark_current_a and monitor_tick_row'},
  {id:'drop', name:'drop-port power', symbol:String.raw`P_{\mathrm{drop}}`, unit:'W', position:[2,2],
    question:'What makes the drop peak and through-port dip?',
    meaning:'constructive circulation raises the drop power near resonance. The through port has a dip as its direct and ring-coupled fields interfere.',
    equation:String.raw`\frac{P_{\mathrm{drop}}}{P_{\mathrm{in}}}=\frac{\kappa_t^2\kappa_d^2a}{1-2r\cos\Phi+r^2}`,
    extra:String.raw`r=\sqrt{1-\kappa_t^2}\sqrt{1-\kappa_d^2}\,a`,
    note:'this Airy expression is for one laser line and data level without backscatter. The engine also handles both data levels, multiple laser lines, and backscatter.',
    definitions:[['P_{\\mathrm{drop}}','drop-port power','W'],['\\kappa_t^2','through-coupler power fraction','1'],['\\kappa_d^2','drop-coupler power fraction','1'],['a','field amplitude retained per round trip','1'],['r','field amplitude retained around the complete feedback loop','1'],['\\Phi','total round-trip phase','rad'],['P_{\\mathrm{in}}','incident optical power','W']],
    records:['selfheat.r_drop_sum','ring_float.p_pd'], source:'mrm/plant/optics.py · _ring_without_backscatter and bus_propagate'},
];

export const LINKS = [
  {from:'heater',to:'temperature',ports:['right','left']},
  {from:'temperature',to:'resonance',ports:['right','left']},
  {from:'resonance',to:'phase',ports:['bottom','top']},
  {from:'phase',to:'loss',ports:['left','right']},
  {from:'loss',to:'absorption',ports:['left','right']},
  {from:'absorption',to:'temperature',ports:['top','bottom'],feedback:true},
  {from:'phase',to:'drop',ports:['bottom','top']},
  {from:'drop',to:'current',ports:['left','right']},
];

const ENGINE_SYMBOLS = /^(?:cfg\.array\.n_rings$|pp\.n_rings$|cfg\.sim\.|cfg\.controller_interface\.|pp\.timing\.|P\.dt$|P\.n_rings$)/;
export function isPhysics(n) {
  if (!n.symbol || n.kind === 'unused' || ENGINE_SYMBOLS.test(n.id)) return false;
  if (['controller','uncertainty','metric'].includes(n.type)) return false;
  if (/^(?:code|bool|tick|ticks|count|index|enum|mixed)$/.test(n.unit || '') && !/ring_int\.order|optics\.order$/.test(n.id)) return false;
  return true;
}

export function topic(n) {
  const text=[n.id,n.group,n.meaning || ''].join(' ').toLowerCase();
  if (/monitor|sensor|sensing|responsiv|dark|photodiode|adc|noise|pd_/.test(text)) return 'measurement';
  if (/thermal|thermo|temperature|tau_fast|tau_slow|th_[a-z]|t_ring|t_ref|selfheat\.tref/.test(text)) return 'thermal';
  if (/heater|junction|bias|driver|electrical|tcr|capacitance|depletion/.test(text)) return 'electrical';
  return 'optics';
}

export const TOPICS = {optics:'optics & geometry', thermal:'heat & temperature', electrical:'electrical drive', measurement:'photodiodes & measurement'};
export const CONTEXTS = {leaf:'preset quantity', setting:'preset setting', derived:'derived quantity', plantparams:'fixed parameter', kparam:'kernel parameter', state:'runtime quantity', truth:'reported quantity', channel:'interface channel', metric:'reported statistic', uncertainty:'uncertainty setting', controller:'controller'};

// Presentation definitions fill sparse source comments without inventing symbols for code fields.
const TERMS = new Map([
  ['R_{H,0}','heater resistance at its reference temperature','ohm'],
  [String.raw`\alpha_L`,'fractional ring-length expansion per kelvin','1/K'],
  [String.raw`\partial n_{\mathrm{eff}}/\partial T`,'effective-index temperature sensitivity','1/K'],
  ['s_0','thermal tuning slope at its anchor temperature','m/K'], ['T_0','thermal tuning anchor temperature','K'],
  [String.raw`I_{\mathrm{sup,max}}`,'total heater supply current limit','A'],
  [String.raw`P_{\mathrm{sup,max}}`,'total heater supply power limit','W'],
  ['P_{L,0}','reference laser power per line at the module output','W'],
  ['f_{L,0}','laser line frequency including its static offset','Hz'],
  [String.raw`\alpha_{\mathrm{bus}}`,'bus power attenuation per length between rings','1/m'],
  ['R','ring radius used by the phase model','m'],
  [String.raw`T_{\mathrm{ref}}`,'cold-resonance reference temperature','K'],
  [String.raw`V_{\mathrm{ref}}`,'cold-resonance reference bias','V'],
  [String.raw`\delta_0`,'cold offset, laser wavelength minus resonance wavelength','m'],
  [String.raw`\Delta f_{\mathrm{FWHM}}`,'loaded resonance linewidth in frequency','Hz'],
  [String.raw`\Delta\lambda_{\mathrm{FWHM}}`,'loaded resonance linewidth in wavelength','m'],
  [String.raw`\Delta f_{\mathrm{split}}`,'frequency splitting of the clockwise and counterclockwise modes','Hz'],
  [String.raw`\mathrm{d}n_g/\mathrm{d}\lambda`,'group-index wavelength sensitivity','1/m'],
  [String.raw`R_{\mathrm{th}}`,'steady ring-temperature rise per heater watt','K/W'],
  [String.raw`R_{\mathrm{th,opt}}`,'steady ring-temperature rise per absorbed optical watt','K/W'],
  [String.raw`s(T_{\mathrm{lock}})\,R_{\mathrm{th,opt}}`,'resonance tuning per absorbed optical watt at lock','m/W'],
  [String.raw`\eta_{\mathrm{tune}}`,'resonance tuning per heater watt from the FSR measurement','m/W'],
  [String.raw`T_{\mathrm{lock}}`,'ring temperature at lock under the reference ambient condition','K'],
  [String.raw`V_{\mathrm{comp}}`,'maximum voltage available to the heater current drive','V'],
  [String.raw`\left.\mathrm{d}\lambda_r/\mathrm{d}V\right\vert_{0\,\mathrm{V}}`,'resonance sensitivity to junction bias at zero volts','m/V'],
  [String.raw`\mathrm{OMA}_{\mathrm{sens}}`,'receiver optical modulation amplitude required at the reference bit error ratio','W'],
  [String.raw`\mathrm{d}f_L/\mathrm{d}t`,'laser frequency drift rate','Hz/s'],
  ['f_L','current laser frequency per line','Hz'], ['P_L','current laser power per line at the module output','W'],
  [String.raw`P_{L,\mathrm{chip}}`,'laser power per line reaching the chip input','W'],
  ['V_1 - V_0','junction voltage swing between the two data levels','V'],
  ['V_0','junction reverse voltage at data level zero','V'], ['V_1','junction reverse voltage at data level one','V'],
  ['C_j','charge-equivalent junction capacitance over the data swing','F'],
  ['a(V_0)','round-trip field transmission at data level zero','1'], ['a(V_1)','round-trip field transmission at data level one','1'],
  [String.raw`\eta_{\mathrm{heat}}(V_0)`,'absorbed fraction of intrinsic loss at data level zero','1'],
  [String.raw`\eta_{\mathrm{heat}}(V_1)`,'absorbed fraction of intrinsic loss at data level one','1'],
  [String.raw`\phi_{\mathrm{ref}}(V_0)`,'reference round-trip propagation phase at data level zero','rad'],
  [String.raw`\phi_{\mathrm{ref}}(V_1)`,'reference round-trip propagation phase at data level one','rad'],
  [String.raw`\kappa_d^2(\lambda_L)`,'drop-coupler power fraction at the laser wavelength','1'],
  [String.raw`\kappa_t^2(\lambda_L)`,'through-coupler power fraction at the laser wavelength','1'],
  [String.raw`\tau_{\mathrm{rt}}(\lambda_L)`,'round-trip group delay at the laser wavelength','s'],
  [String.raw`\mathrm{d}\phi/\mathrm{d}\lambda_r`,'propagation-phase sensitivity to resonance wavelength','rad/m'],
  [String.raw`P_{\mathrm{bus,in}}`,'optical power per line entering the ring bus','W'],
  [String.raw`P_{\mathrm{bus,out}}`,'data-averaged optical power per line leaving the ring bus','W'],
  [String.raw`P_{\mathrm{in}}`,'optical power arriving at the ring input','W'],
  [String.raw`P_{\mathrm{drv}}`,'driver heat from data activity','W'],
  [String.raw`P_{\mathrm{junc}}`,'junction heat from data activity','W'],
  [String.raw`T_{\mathrm{base}}`,'base temperature including processor heating','K'],
  ['I_H','heater current after the supply limit','A'], ['V_H','heater voltage after the supply limit','V'],
  ['R_H','heater resistance at its current temperature','ohm'], ['T_H','heater-resistor temperature','K'],
  ['P_0','bus-output power with this ring at data level zero','W'], ['P_1','bus-output power with this ring at data level one','W'],
  [String.raw`P_{\mathrm{thru}}(V_0)`,'through-port power at data level zero','W'],
  [String.raw`P_{\mathrm{thru}}(V_1)`,'through-port power at data level one','W'],
  [String.raw`I_{\mathrm{dark}}`,'photodiode current without incident light','A'],
  [String.raw`T_{\mathrm{ADC}}`,'temperature of the monitor ADC reference','K'],
  [String.raw`T_{\mathrm{PD}}`,'monitor photodiode temperature','K'],
  [String.raw`\lambda_r - \lambda_L`,'detuning, resonance wavelength minus laser wavelength','m'],
  [String.raw`\lambda_r`,'resonance wavelength nearest the assigned laser line','m'],
  [String.raw`I_{\mathrm{mon}}`,'monitor photocurrent including dark current','A'],
  [String.raw`\mathrm{OMA}`,'difference between the two bus-output optical power levels','W'],
  [String.raw`\mathrm{ER}`,'through-port extinction ratio between data levels','dB'],
  [String.raw`\mathrm{IL}`,'through-port insertion loss at the higher power level','dB'],
  [String.raw`T_{\mathrm{ring}}`,'ring temperature','K'],
  [String.raw`P_{\mathrm{abs}}`,'absorbed optical power','W'],
  [String.raw`P_{\mathrm{drop}}`,'drop-port optical power','W'],
  ['P_H','heater power','W'],
  ['s(T_{\\mathrm{ring}})','thermal tuning slope at the current reference temperature','m/K'],
  [String.raw`n_{\mathrm{eff}}`,'effective refractive index','1'],
  [String.raw`n_{\mathrm{eff,phase}}`,'phase-consistent effective index','1'],
  [String.raw`n_{\mathrm{eff,phase}} - n_{\mathrm{eff}}`,'effective-index correction used to match the resonance order','1'],
  ['n_g','group index','1'], ['Q_L','loaded quality factor','1'],
  [String.raw`Q_{\mathrm{int}}`,'intrinsic quality factor','1'],['Q_t','through-coupler quality factor','1'],['Q_d','drop-coupler quality factor','1'],
  [String.raw`\kappa_t^2`,'through-coupler power fraction','1'],[String.raw`\kappa_d^2`,'drop-coupler power fraction','1'],
  ['a','round-trip field transmission','1'], ['m','longitudinal resonance order','1'], ['m_j','junction depletion exponent','1'],
  ['x_d','drop decay rate divided by total energy decay rate','1'],[String.raw`1 - \Gamma`,'effective-index sensitivity to the oxide index','1'],
  [String.raw`\Delta f_{\mathrm{FSR}}`,'free spectral range in frequency','Hz'],
  [String.raw`\Delta\lambda_{\mathrm{FSR}}`,'free spectral range in wavelength','m'],
  [String.raw`\Delta\lambda_{\mathrm{FSR,meas}}`,'wavelength span used for the heater-power anchor','m'],
  [String.raw`\tau_{\mathrm{rt}}`,'round-trip group delay','s'],['L','ring round-trip length','m'],
  [String.raw`\lambda_L`,'laser wavelength','m'],[String.raw`\lambda_{r0}`,'cold reference resonance wavelength','m'],
  [String.raw`\delta_0 \bmod \Delta\lambda_{\mathrm{FSR}}`,'cold resonance offset within one free spectral range','m'],
  [String.raw`\alpha_{\mathrm{int}}`,'intrinsic power attenuation per length','1/m'],
  [String.raw`\alpha_{\mathrm{heat}}`,'absorbing power attenuation per length','1/m'],
  [String.raw`\alpha_{\mathrm{nonheat}}`,'non-absorbing power attenuation per length','1/m'],
  [String.raw`\eta_{\mathrm{heat}}`,'absorbed fraction of intrinsic optical loss','1'],
  [String.raw`\mathrm{d}\alpha_L/\mathrm{d}T`,'temperature derivative of fractional length expansion','1/K^2'],
  [String.raw`\partial^2 n_{\mathrm{eff}}/\partial T^2`,'temperature curvature of effective index','1/K^2'],
  [String.raw`\beta_T`,'relative temperature curvature of the thermal tuning slope','1/K'],
  [String.raw`\Delta T_{\mathrm{FSR}}`,'temperature rise required to tune one free spectral range','K'],
  ['s(T_{\\mathrm{lock}})','thermal tuning slope at lock','m/K'],
  [String.raw`R_{\mathrm{th},H}/R_{\mathrm{th}}`,'heater self-heating response divided by heater-to-ring response','1'],
  [String.raw`R_{\mathrm{th,opt}}/R_{\mathrm{th}}`,'optical-to-ring thermal response divided by heater-to-ring response','1'],
  [String.raw`\partial\ln\alpha/\partial T_H`,'relative loss sensitivity to heater temperature','1/K'],
  [String.raw`\partial\ln\alpha/\partial T_{\mathrm{ring}}`,'relative loss sensitivity to ring temperature','1/K'],
  [String.raw`\mathrm{d}\ln\kappa_t^2/\mathrm{d}\lambda`,'relative through-coupling wavelength sensitivity','1/m'],
  [String.raw`\mathrm{d}\ln\kappa_d^2/\mathrm{d}\lambda`,'relative drop-coupling wavelength sensitivity','1/m'],
  [String.raw`C_j(0\,\mathrm{V})`,'junction capacitance at zero bias','F'], ['R_s','junction series resistance','ohm'],
  [String.raw`V_{\mathrm{bi}}`,'built-in junction voltage','V'],
  [String.raw`I_{H,\mathrm{FS}}`,'heater full-scale current','A'],[String.raw`V_{H,\mathrm{FS}}`,'heater full-scale voltage','V'],
  [String.raw`P_{H,\mathrm{FS}}`,'heater full-scale power','W'],[String.raw`P_{\mathrm{FSR}}`,'heater power for one free spectral range','W'],
  ['T_{H,0}','reference heater temperature','K'],[String.raw`\alpha_{\mathrm{TCR}}`,'heater resistance temperature coefficient','1/K'],
  [String.raw`V_{\mathrm{bias,max}}`,'maximum bias voltage','V'],[String.raw`V_{\mathrm{bias,min}}`,'minimum bias voltage','V'],
  [String.raw`I_{\mathrm{ADC,FS}}`,'ADC full-scale input current','A'],['E_a','dark-current activation energy','J'],
  [String.raw`I_{\mathrm{dark},0}`,'dark current at its reference temperature','A'],
  [String.raw`T_{\mathrm{dark},0}`,'reference temperature for dark current','K'],
  [String.raw`f_{\mathrm{corner}}`,'flicker-noise corner frequency','Hz'],['i_n','input current-noise amplitude density','A/sqrt(Hz)'],
  [String.raw`\mathcal{R}`,'photodiode responsivity','A/W'],['R_b','bit rate','1/s'],
  [String.raw`E_{\mathrm{tr}}`,'driver energy per transition','J'],[String.raw`R_{\mathrm{out}}`,'driver output resistance','ohm'],
  [String.raw`P_{\mathrm{drv,static}}`,'static driver power','W'],[String.raw`V_{\mathrm{pp}}`,'peak-to-peak data voltage','V'],
  [String.raw`\Delta f_{\mathrm{laser}}`,'laser linewidth','Hz'],[String.raw`\mathrm{RIN}`,'relative-intensity-noise power density','1/Hz'],
  [String.raw`\mathrm{BER}`,'bit error ratio','1'],[String.raw`\mathrm{relOMA}`,'optical modulation amplitude divided by input power','1'],
  [String.raw`2(\lambda_r - \lambda_L)/\Delta\lambda_{\mathrm{FWHM}}`,'detuning measured in half-linewidths','1'],
  [String.raw`T_{\mathrm{amb}}`,'ambient temperature','K'],[String.raw`V_{\mathrm{bias}}`,'junction reverse bias','V'],
].map(([symbol,name,unit])=>[symbol,{name,unit}]));

const SUFFIX_UNITS = [['_a_per_rt_hz','A/sqrt(Hz)'],['_pa_per_rt_hz','pA/sqrt(Hz)'],['_pm_per_k','pm/K'],['_m_per_k','m/K'],['_per_k2','1/K^2'],['_per_k','1/K'],['_nm_per_mw','nm/mW'],['_k_per_mw','K/mW'],['_m_per_w','m/W'],['_a_per_w','A/W'],['_per_m','1/m'],['_ghz','GHz'],['_thz','THz'],['_nm','nm'],['_um','um'],['_pm','pm'],['_hz','Hz'],['_dbm','dBm'],['_db','dB'],['_ff','fF'],['_fj','fJ'],['_na','nA'],['_ua','uA'],['_ohm','ohm'],['_ev','eV'],['_c','degC'],['_k','K'],['_m','m'],['_s','s'],['_v','V'],['_a','A'],['_w','W'],['_j','J'],['_f','F']];
const UNIT_LABELS = {a:'A',w:'W',thz:'THz',ghz:'GHz',gbps:'Gbit/s',fj:'fJ',mw:'mW',v:'V',mhz:'MHz',dbm:'dBm',db_per_hz:'dB/Hz',db_per_mm:'dB/mm',c:'degC',per_k:'1/K',ff:'fF',pm_per_v:'pm/V',ua:'uA',hz:'Hz',ev:'eV',na:'nA',a_per_w:'A/W',per_k2:'1/K^2',per_nm:'1/nm',db_per_cm:'dB/cm',pm_per_k:'pm/K',frac:'1',nm_per_mw:'nm/mW',k_per_mw:'K/mW',index:'1'};
export function quantity(n) {
  const symbol=normalizeTex(n.symbol), known=TERMS.get(symbol);
  const suffix=SUFFIX_UNITS.find(([suffix])=>n.id.endsWith(suffix));
  const statedUnit=n.unit && n.unit!=='SI' ? n.unit : '';
  // Compound current-noise units take precedence over a stale generic Hz suffix.
  const noiseUnit=/_a_per_rt_hz$|_pa_per_rt_hz$/.test(n.id) ? suffix?.[1] : '';
  const rawUnits=n.id==='pp.junction.depletion_a_m' ? 'm/V^m_j' : noiseUnit || statedUnit || suffix?.[1] || known?.unit || '1';
  const units=UNIT_LABELS[rawUnits] || rawUnits;
  const raw=(n.meaning || '').replace(/^\([^)]*\)\s*/,'').trim();
  const sparse=!raw || /^\([^)]*\)$/.test(raw) || raw.length<5 || /^(?:\(0 when absent|\(nan when|\(R\d)/.test(raw);
  const meaning=n.id==='pp.junction.depletion_a_m' ? 'coefficient in the analytic resonance-shift law, with units set by the depletion exponent' : sparse && known ? known.name : raw || known?.name || n.symbol_text || symbol;
  return {name:n.id==='pp.junction.depletion_a_m' ? 'analytic depletion-shift coefficient' : known?.name || meaning,meaning,unit:units};
}


const LOCAL_NOTATION = {
  'selfheat.beta': {symbol:String.raw`\beta_{T,\mathrm{ref}}`,meaning:'relative curvature of the thermal tuning law expanded about the reference temperature',unit:'1/K'},
  'S.th_x': {symbol:'x_j',meaning:'temperature rise stored in thermal path j',unit:'K'},
  'S.th_t_obs': {symbol:'T_o',meaning:'absolute temperature at thermal observer o',unit:'K'},
};
export function withNotation(n) {
  return !n.symbol && LOCAL_NOTATION[n.id] ? {...n,...LOCAL_NOTATION[n.id],notation_source:'defined in the physics reference'} : n;
}
export function recordKind(n) {
  if (isPhysics(n)) return 'physical quantity';
  const physicalUnit=/^(?:m|K|W|V|A|J|F|Hz|rad|ohm|1\/K|W\/K|K\/W)$/;
  if(physicalUnit.test(n.unit||'') && n.kind!=='unused' && !ENGINE_SYMBOLS.test(n.id))return 'physical value · code representation';
  return 'programming / simulation';
}
