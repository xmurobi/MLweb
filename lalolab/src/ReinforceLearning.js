/**
 * Created by robi on 17/8/21.
 */

// syntactic sugar function for getting default parameter values
var getopt = function(opt, field_name, default_value) {
    if(typeof opt === 'undefined') { return default_value; }
    return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
};

var setConst = function(arr, c) {
    for(var i=0,n=arr.length;i<n;i++) {
        arr[i] = c;
    }
};

var sampleWeighted = function(p) {
    var r = Math.random();
    var c = 0.0;
    for(var i=0,n=p.length;i<n;i++) {
        c += p[i];
        if(c >= r) { return i; }
    }

    // wtf
    return -1;
};

//////////////////////////////////////////////////
// Generic class for RL(implements DP by default)
// params.env.getNumStates()
//      returns an integer of total number of states
// params.env.getMaxNumActions()
//      returns an integer with max number of actions in any state
// params.env.allowedActions(s)
//      takes an integer s and returns a list of available actions,
//      which should be integers from zero to maxNumActions
// params.env.nextStateDistribution(s,a)
//      which is a misnomer, since right now the library assumes
//      deterministic MDPs that have a single unique new state for
//      every (state, action) pair. Therefore, the function should
//      return a single integer that identifies the next state of the world
// params.env.reward(s,a,ns)
//      which returns a float for the reward achieved by the agent
//      for the s, a, ns transition. In the simplest case, the reward
//      is usually based only the state s.
///////////////////////////////////////////////////
/**
 * @constructor
 */
function ReinforceLearning (algorithm, params ) {

    if ( typeof(algorithm) == "undefined" ) {
        var algorithm = DP;
    }
    else if (typeof(algorithm) == "string")
        algorithm = eval(algorithm);

    this.type = "ReinforceLearning:" + algorithm.name;

    this.algorithm = algorithm.name;
    this.userParameters = params;

    // Functions that depend on the algorithm:
    this.construct = algorithm.prototype.construct;
    this.train = algorithm.prototype.train;
    this.act = algorithm.prototype.act;

    if (  algorithm.prototype.actTrain )
        this.actTrain = algorithm.prototype.actTrain;
    if (  algorithm.prototype.plan )
        this.plan = algorithm.prototype.plan;
    if (  algorithm.prototype.evaluatePolicy )
        this.evaluatePolicy = algorithm.prototype.evaluatePolicy;
    if (  algorithm.prototype.updatePolicy )
        this.updatePolicy = algorithm.prototype.updatePolicy;
    if ( algorithm.prototype.updateModel )
        this.updateModel = algorithm.prototype.updateModel;
    if ( algorithm.prototype.updatePriority )
        this.updatePriority = algorithm.prototype.updatePriority;
    if ( algorithm.prototype.learnFromTuple )
        this.learnFromTuple = algorithm.prototype.learnFromTuple;

    if (  algorithm.prototype.reset )
        this.reset = algorithm.prototype.reset;
    if (  algorithm.prototype.resetEpisode )
        this.resetEpisode = algorithm.prototype.resetEpisode;


    // Initialization depending on algorithm
    this.construct(params);
};

ReinforceLearning.prototype.construct = function ( params ) {
    // Read this.params and create the required fields for a specific algorithm
    // Set parameters:
    var i;
    if ( params) {
        for (i in params)
            this[i] = params[i];
    }
};

//////////////////////////////////////////////////
// Copy from karpathy's reinforcejs:
// http://cs.stanford.edu/people/karpathy/reinforcejs/index.html
// DPAgent performs Value Iteration
// - can also be used for Policy Iteration if you really wanted to
// - requires model of the environment :(
// - does not learn from experience :(
// - assumes finite MDP :(
//////////////////////////////////////////////////
function DP (params) {
    var that = new ReinforceLearning ( DP, params);
    return that;
};

DP.prototype.construct = function (params) {
    this.V = params.V || null; // state value function
    this.P = params.P || null; // policy distribution \pi(s,a)
    this.env = params.env; // store pointer to environment
    this.gamma = getopt(params, 'gamma', 0.75); // future reward discount factor
    this.error = getopt(params, 'error', 0.0001);// error for convergence
    this.ns = getopt(params, 'ns', this.env.getNumStates());
    this.na = getopt(params, 'na', this.env.getMaxNumActions());
    this.converged = getopt(params, 'converged', false);

    this.reset();
};

DP.prototype.reset = function (force) {
    if (this.converged && !force) return;

    // reset the agent's policy and value function
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();
    this.V = zeros(this.ns);
    this.P = zeros(this.ns * this.na);
    // initialize uniform random policy
    for(var s=0;s<this.ns;s++) {
        var poss = this.env.allowedActions(s);
        for(var i=0,n=poss.length;i<n;i++) {
            this.P[poss[i]*this.ns+s] = 1.0 / poss.length;
        }
    }
};

DP.prototype.train = function () {
    while (!this.converged) {
        // perform a single round of value iteration
        this.evaluatePolicy(); // writes this.V
        this.updatePolicy(); // writes this.P
    }
};

DP.prototype.evaluatePolicy = function () {
    // perform a synchronous update of the value function
    var Vnew = zeros(this.ns);
    // converge counter
    var cc = 0;
    for(var s=0;s<this.ns;s++) {
        // integrate over actions in a stochastic policy
        // note that we assume that policy probability mass over allowed actions sums to one
        var v = 0.0;
        var poss = this.env.allowedActions(s);
        for(var i=0,n=poss.length;i<n;i++) {
            var a = poss[i];
            var prob = this.P[a*this.ns+s]; // probability of taking action under policy
            if(prob === 0) { continue; } // no contribution, skip for speed
            var ns = this.env.nextStateDistribution(s,a);
            var rs = this.env.reward(s,a,ns); // reward for s->a->ns transition
            v += prob * (rs + this.gamma * this.V[ns]);
        }
        Vnew[s] = v;
        if (Math.abs(Vnew[s] - this.V[s]) < this.error) cc++;
    }
    this.V = Vnew; // swap
    this.converged = cc >= this.ns;
};

DP.prototype.updatePolicy = function () {
    // update policy to be greedy w.r.t. learned Value function
    for(var s=0;s<this.ns;s++) {
        var poss = this.env.allowedActions(s);
        // compute value of taking each allowed action
        var vmax, nmax;
        var vs = [];
        for(var i=0,n=poss.length;i<n;i++) {
            var a = poss[i];
            var ns = this.env.nextStateDistribution(s,a);
            var rs = this.env.reward(s,a,ns);
            var v = rs + this.gamma * this.V[ns];
            vs.push(v);
            if(i === 0 || v > vmax) { vmax = v; nmax = 1; }
            else if(v === vmax) { nmax += 1; }
        }
        // update policy smoothly across all argmaxy actions
        for(var i=0,n=poss.length;i<n;i++) {
            var a = poss[i];
            this.P[a*this.ns+s] = (vs[i] === vmax) ? 1.0/nmax : 0.0;
        }
    }
};

DP.prototype.actTrain = function (s) {
    // behave according to the learned policy
    var poss = this.env.allowedActions(s);
    var ps = [];
    for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        var prob = this.P[a*this.ns+s];
        ps.push(prob);
    }
    var maxi = sampleWeighted(ps);
    return poss[maxi];
};

DP.prototype.act = function (s) {
    var poss = this.env.allowedActions(s);
    var ps = [];
    for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        var prob = this.P[a*this.ns+s];
        ps.push(prob);
    }
    var maxi = findmax(ps);
    return poss[maxi];
};

//////////////////////////////////////////////////
// Copy from karpathy's reinforcejs:
// http://cs.stanford.edu/people/karpathy/reinforcejs/index.html
// QAgent uses TD (Q-Learning, SARSA)
// - does not require environment model :)
// - learns from experience :)
//////////////////////////////////////////////////
function TD (params) {
    var that = new ReinforceLearning ( TD, params);
    return that;
};

TD.prototype.construct = function (params) {
    this.update = getopt(params, 'update', 'qlearn'); // qlearn | sarsa
    this.gamma = getopt(params, 'gamma', 0.75); // future reward discount factor
    this.epsilon = getopt(params, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(params, 'alpha', 0.01); // value function learning rate
    this.error = getopt(params, 'error', 0.0001);// error for convergence

    // class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
    this.smooth_policy_update = getopt(params, 'smooth_policy_update', false);
    this.beta = getopt(params, 'beta', 0.01); // learning rate for policy, if smooth updates are on

    // eligibility traces
    this.lambda = getopt(params, 'lambda', 0); // eligibility trace decay. 0 = no eligibility traces used
    this.replacing_traces = getopt(params, 'replacing_traces', true);

    // optional optimistic initial values
    this.q_init_val = getopt(params, 'q_init_val', 0);

    this.planN = getopt(params, 'planN', 0); // number of planning steps per learning iteration (0 = no planning)
    this.converged = getopt(params, 'converged', false);

    this.Q = params.Q || null; // state action value function
    this.P = params.P || null; // policy distribution \pi(s,a)
    this.e = params.e || null; // eligibility trace
    this.pq = params.pq || null;
    this.env_model_s = params.env_model_s || null; // environment model (s,a) -> (s',r)
    this.env_model_r = params.env_model_r || null; // environment model (s,a) -> (s',r)
    this.env = params.env; // store pointer to environment
    this.ns = getopt(params, 'ns', this.env.getNumStates());
    this.na = getopt(params, 'na', this.env.getMaxNumActions());

    this.reset();
};

TD.prototype.reset = function (force) {
    // agent memory, needed for streaming updates
    // (s0,a0,r0,s1,a1,r1,...)
    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;

    if (this.converged && !force) return;

    // reset the agent's policy and value function
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();
    this.Q = zeros(this.ns * this.na);
    if(this.q_init_val !== 0) { setConst(this.Q, this.q_init_val); }
    this.P = zeros(this.ns * this.na);
    this.e = zeros(this.ns * this.na);

    // model/planning vars
    this.env_model_s = zeros(this.ns * this.na);
    setConst(this.env_model_s, -1); // init to -1 so we can test if we saw the state before
    this.env_model_r = zeros(this.ns * this.na);
    this.sa_seen = [];
    this.pq = zeros(this.ns * this.na);

    // initialize uniform random policy
    for(var s=0;s<this.ns;s++) {
        var poss = this.env.allowedActions(s);
        for(var i=0,n=poss.length;i<n;i++) {
            this.P[poss[i]*this.ns+s] = 1.0 / poss.length;
        }
    }
};

/**
 * Training in the environment:
 * var a = agent.actTrain(state); // ask agent for an action
 * var obs = env.sampleNextState(state, a); // run it through environment dynamics
 * agent.train(obs.r); // allow opportunity for the agent to learn
 * state = obs.ns; // evolve environment to next state
 * if(typeof obs.reset_episode !== 'undefined') {
 *  agent.resetEpisode();
 *  // converged?
 *  if(agent.converged) {
 *      // do something...
 *  } else {
 *      // goto next loop and continue training...
 *  }
 * }
 * @param r1    reward for s1a1
 */
TD.prototype.train = function (r1) {
    // takes reward for previous action, which came from a call to act()
    if(!(this.r0 == null)) {
        this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1, this.lambda);
        if(this.planN > 0) {
            this.updateModel(this.s0, this.a0, this.r0, this.s1);
            this.plan();
        }
    }
    this.r0 = r1; // store this for next update
};

TD.prototype.resetEpisode = function() {
    if (this.converged) return;
    // an episode finished
    // check if converged
    var l = this.na * this.ns;
    if (typeof this.lastQ === 'undefined') {
        this.lastQ = this.Q.slice(0);
    } else {
        var cc = 0;
        for (var i = 0 ;i < l; ++i) {
            if (Math.abs(this.lastQ[i]-this.Q[i]) < this.error) ++cc;
        }
        this.converged = cc >= l;

        this.lastQ = this.Q.slice(0);
    }
};

TD.prototype.plan = function () {
    // order the states based on current priority queue information
    var spq = [];
    for(var i=0,n=this.sa_seen.length;i<n;i++) {
        var sa = this.sa_seen[i];
        var sap = this.pq[sa];
        if(sap > 1e-5) { // gain a bit of efficiency
            spq.push({sa:sa, p:sap});
        }
    }
    spq.sort(function(a,b){ return a.p < b.p ? 1 : -1});

    // perform the updates
    var nsteps = Math.min(this.planN, spq.length);
    for(var k=0;k<nsteps;k++) {
        // random exploration
        //var i = randi(0, this.sa_seen.length); // pick random prev seen state action
        //var s0a0 = this.sa_seen[i];
        var s0a0 = spq[k].sa;
        this.pq[s0a0] = 0; // erase priority, since we're backing up this state
        var s0 = s0a0 % this.ns;
        var a0 = Math.floor(s0a0 / this.ns);
        var r0 = this.env_model_r[s0a0];
        var s1 = this.env_model_s[s0a0];
        var a1 = -1; // not used for Q learning
        if(this.update === 'sarsa') {
            // generate random action?...
            var poss = this.env.allowedActions(s1);
            var a1 = poss[randi(0,poss.length)];
        }
        this.learnFromTuple(s0, a0, r0, s1, a1, 0); // note lambda = 0 - shouldnt use eligibility trace here
    }

};

TD.prototype.learnFromTuple = function (s0, a0, r0, s1, a1, lambda) {
    var sa = a0 * this.ns + s0;

    // calculate the target for Q(s,a)
    if(this.update === 'qlearn') {
        // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
        var poss = this.env.allowedActions(s1);
        var qmax = 0;
        for(var i=0,n=poss.length;i<n;i++) {
            var s1a = poss[i] * this.ns + s1;
            var qval = this.Q[s1a];
            if(i === 0 || qval > qmax) { qmax = qval; }
        }
        var target = r0 + this.gamma * qmax;
    } else if(this.update === 'sarsa') {
        // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
        var s1a1 = a1 * this.ns + s1;
        var target = r0 + this.gamma * this.Q[s1a1];
    }

    if(lambda > 0) {
        // perform an eligibility trace update
        if(this.replacing_traces) {
            this.e[sa] = 1;
        } else {
            this.e[sa] += 1;
        }
        var edecay = lambda * this.gamma;
        var state_update = zeros(this.ns);
        for(var s=0;s<this.ns;s++) {
            var poss = this.env.allowedActions(s);
            for(var i=0;i<poss.length;i++) {
                var a = poss[i];
                var saloop = a * this.ns + s;
                var esa = this.e[saloop];
                var update = this.alpha * esa * (target - this.Q[saloop]);
                this.Q[saloop] += update;
                this.updatePriority(s, a, update);
                this.e[saloop] *= edecay;
                var u = Math.abs(update);
                if(u > state_update[s]) { state_update[s] = u; }
            }
        }
        for(var s=0;s<this.ns;s++) {
            if(state_update[s] > 1e-5) { // save efficiency here
                this.updatePolicy(s);
            }
        }
        if(this.explored && this.update === 'qlearn') {
            // have to wipe the trace since q learning is off-policy :(
            this.e = zeros(this.ns * this.na);
        }
    } else {
        // simpler and faster update without eligibility trace
        // update Q[sa] towards it with some step size
        var update = this.alpha * (target - this.Q[sa]);
        this.Q[sa] += update;
        this.updatePriority(s0, a0, update);
        // update the policy to reflect the change (if appropriate)
        this.updatePolicy(s0);
    }

};

TD.prototype.updateModel = function (s0, a0, r0, s1) {
    // transition (s0,a0) -> (r0,s1) was observed. Update environment model
    var sa = a0 * this.ns + s0;
    if(this.env_model_s[sa] === -1) {
        // first time we see this state action
        this.sa_seen.push(a0 * this.ns + s0); // add as seen state
    }
    this.env_model_s[sa] = s1;
    this.env_model_r[sa] = r0;
};

TD.prototype.updatePriority = function (s,a,u) {
    // used in planning. Invoked when Q[sa] += update
    // we should find all states that lead to (s,a) and upgrade their priority
    // of being update in the next planning step
    u = Math.abs(u);
    if(u < 1e-5) { return; } // for efficiency skip small updates
    if(this.planN === 0) { return; } // there is no planning to be done, skip.
    for(var si=0;si<this.ns;si++) {
        // note we are also iterating over impossible actions at all states,
        // but this should be okay because their env_model_s should simply be -1
        // as initialized, so they will never be predicted to point to any state
        // because they will never be observed, and hence never be added to the model
        for(var ai=0;ai<this.na;ai++) {
            var siai = ai * this.ns + si;
            if(this.env_model_s[siai] === s) {
                // this state leads to s, add it to priority queue
                this.pq[siai] += u;
            }
        }
    }
};

TD.prototype.updatePolicy = function (s) {
    var poss = this.env.allowedActions(s);
    // set policy at s to be the action that achieves max_a Q(s,a)
    // first find the maxy Q values
    var qmax, nmax;
    var qs = [];
    for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        var qval = this.Q[a*this.ns+s];
        qs.push(qval);
        if(i === 0 || qval > qmax) { qmax = qval; nmax = 1; }
        else if(qval === qmax) { nmax += 1; }
    }
    // now update the policy smoothly towards the argmaxy actions
    var psum = 0.0;
    for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        var target = (qs[i] === qmax) ? 1.0/nmax : 0.0;
        var ix = a*this.ns+s;
        if(this.smooth_policy_update) {
            // slightly hacky :p
            this.P[ix] += this.beta * (target - this.P[ix]);
            psum += this.P[ix];
        } else {
            // set hard target
            this.P[ix] = target;
        }
    }
    if(this.smooth_policy_update) {
        // renomalize P if we're using smooth policy updates
        for(var i=0,n=poss.length;i<n;i++) {
            var a = poss[i];
            this.P[a*this.ns+s] /= psum;
        }
    }
};

TD.prototype.actTrain = function (s) {
    // act according to epsilon greedy policy
    var poss = this.env.allowedActions(s);
    var probs = [];
    for(var i=0,n=poss.length;i<n;i++) {
        probs.push(this.P[poss[i]*this.ns+s]);
    }
    // epsilon greedy policy
    if(Math.random() < this.epsilon) {
        var a = poss[randi(0,poss.length)]; // random available action
        this.explored = true;
    } else {
        var a = poss[sampleWeighted(probs)];
        this.explored = false;
    }
    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;
    return a;
};

TD.prototype.act = function (s) {
    // act according to epsilon greedy policy
    var poss = this.env.allowedActions(s);
    var probs = [];
    for(var i=0,n=poss.length;i<n;i++) {
        probs.push(this.P[poss[i]*this.ns+s]);
    }

    var a = poss[findmax(probs)];

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;
    return a;
};