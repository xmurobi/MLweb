/**
 * Created by robi on 17/8/21.
 */

function assert(condition, message) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message; // Fallback
    }
};

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

var maxi = function(w) {
    // argmax of array w
    var maxv = w[0];
    var maxix = 0;
    for(var i=1,n=w.length;i<n;i++) {
        var v = w[i];
        if(v > maxv) {
            maxix = i;
            maxv = v;
        }
    }
    return maxix;
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

    this.construct = algorithm.prototype.construct;
    this.train = algorithm.prototype.train;
    this.act = algorithm.prototype.act;
    this.actTrain = algorithm.prototype.actTrain;
    if (  algorithm.prototype.reset )
        this.reset = algorithm.prototype.reset;
    if (  algorithm.prototype.resetEpisode )
        this.resetEpisode = algorithm.prototype.resetEpisode;

    // Functions that depend on the algorithm:
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

    this.sa_seen = [];
    this.pq = zeros(this.ns * this.na);

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

//////////////////////////////////////////////////
// Deep Q Learning:
// We're going to extend the Temporal Difference Learning (Q-Learning) to continuous state spaces. In the previos demo we've
// estimated the Q learning function Q(s,a) as a lookup table. Now, we are going to use a function approximator to model
// Q(s,a)=fθ(s,a), where θ are some parameters (e.g. weights and biases of a neurons in a network). However, as we will see
// everything else remains exactly the same. The first paper that showed impressive results with this approach was Playing Atari
// with Deep Reinforcement Learning at NIPS workshop in 2013, and more recently the Nature paper Human-level control through
// deep reinforcement learning, both by Mnih et al. However, more impressive than what we'll develop here, their function fθ,a was
// an entire Convolutional Neural Network that looked at the raw pixels of Atari game console. It's hard to get all that to work in JS :(
//
// Recall that in Q-Learning we had training data that is a single chain of
// st,at,rt,st+1,at+1,rt+1,st+2,… where the states ss and the rewards r are samples from the environment
// dynamics , and the actions aa are sampled from the agent's policy at∼π(a∣st). The agent's policy in Q-learning is
// to execute the action that is currently estimated to have the highest value (π(a∣s)=argmaxaQ(s,a),or with a
// probability ϵ to take a random action to ensure some exploration. The Q-Learning update at each time step then had the following form:
//
//      Q(st,at)←Q(st,at)+α[rt+γ * max_aQ(st+1,a)−Q(st,at)]
//
// As mentioned, this equation is describing a regular Stochastic Gradient Descent update with learning rate αα and the loss function at time t:
//
//      Lt=(rt+γ * max_aQ(st+1,a)−Q(st,at))^2
//
// Here y=rt+γmaxaQ(st+1,a) is thought of as a scalar-valued fixed target, while we backpropagate through the neural
// network that produced the prediction fθ=Q(st,at). Note that the loss has a standard L2 norm regression form, and that we
// nudge the parameter vector θ in a way that makes the computed Q(s,a) slightly closer to what it should be (i.e. to satisfy the
// Bellman equation). This softly encourages the constraint that the reward should be properly diffused, in expectation, backwards
// through the environment dynamics and the agent's policy.
//
// In other words, Deep Q Learning is a 1-dimensional regression problem with a vanilla neural network,
// solved with vanilla stochastic gradient descent, except our training data is not fixed but generated by interacting with the environment.
//////////////////////////////////////////////////

// Mat holds a matrix which add specific properties for ANN only
var Mat = function(n,d) {
    // n is number of rows d is number of columns
    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
};

Mat.prototype = {
    get: function(row, col) {
        // slow but careful accessor function
        // we want row-major order
        var ix = (this.d * row) + col;
        assert(ix >= 0 && ix < this.w.length);
        return this.w[ix];
    },
    set: function(row, col, v) {
        // slow but careful accessor function
        var ix = (this.d * row) + col;
        assert(ix >= 0 && ix < this.w.length);
        this.w[ix] = v;
    },
    setFrom: function(arr) {
        for(var i=0,n=arr.length;i<n;i++) {
            this.w[i] = arr[i];
        }
    },
    setColumn: function(m, i) {
        for(var q=0,n=m.w.length;q<n;q++) {
            this.w[(this.d * q) + i] = m.w[q];
        }
    },
    randg: function (mu, std) {
        for(var i=0,n=this.w.length;i<n;i++) {
            this.w[i] = randg(mu, std);
        }
        return this;
    },
    randf: function (lo,hi) {
        for(var i=0,n=this.w.length;i<n;i++) {
            this.w[i] = randf(lo, hi);
        }
        return this;
    },
    fixdw: function (c) {
        for(var i=0,n=this.dw.length;i<n;i++) {
            this.dw[i] = c;
        }
        return this;
    },
    copy: function () {
        var b = new Mat(this.n,this.d);
        b.setFrom(this.w);
        return b;
    },
    update: function (alpha) {
        // updates in place
        for(var i=0,n=this.n*this.d;i<n;i++) {
            if(this.dw[i] !== 0) {
                this.w[i] += - alpha * this.dw[i];
                this.dw[i] = 0;
            }
        }
        return this;
    },
    fromMatrix: function(matrix) {
        this.n = matrix.m;
        this.d = matrix.n;
        this.w = vectorCopy(matrix.val) ;
        this.dw = zeros(this.n * this.d);
        return this;
    },
    toMatrix: function () {
        return new Matrix(this.n, this.d, this.w);
    },
};

/**
 * Constract a new
 * @param ns
 * @param na
 * @param hiddens Array that object structure like:
 * {
 *  n:'num_of_hidden_neurons',
 *  fn:'active function in Graph for this layer'
 * }
 * @constructor
 */
var Net = function (ns, na, hiddens) {
    assert(hiddens.length >= 1);

    this.W = [];
    this.b = [];

    this.W[0] = new Mat(hiddens[0].n, ns).randg(0, 0.01);
    this.b[0] = new Mat(hiddens[0].n, 1).randg(0, 0.01);

    var h = 1;
    while (h < hiddens.length) {
        this.W[h-1] = new Mat(hiddens[h].n, hiddens[h-1].n).randg(0, 0.01);
        this.b[h-1] = new Mat(hiddens[h].n, 1).randg(0, 0.01);
        ++h;
    }

    this.W[h] = new Mat(na, hiddens[hiddens.length-1].n).randg(0, 0.01);
    this.b[h] = new Mat(na, 1).randg(0, 0.01);

    this.hfn = [];
    h = 0;
    while (h < hiddens.length) {
        this.hfn.push(hiddens[h++].fn);
    }
};

Net.prototype = {
    copy: function () {
        var b = new Net();
        for(var p in this) {
            if(this.hasOwnProperty(p)){
                b[p] = [];
                for (var i = 0 ;i < this[p].length; ++i) {
                    if (this[p][i] instanceof Mat)
                        b[p][i] = this[p][i].copy();
                    else
                        b[p][i] = this[p][i];
                }
            }
        }
        return b;
    },

    update: function (alpha) {
        for(var p in this) {
            if(this.hasOwnProperty(p)){
                for (var i = 0 ;i < this[p].length; ++i) {
                    if (this[p][i] instanceof Mat)
                        this[p][i].update(alpha);
                }
            }
        }
    },

    forward: function (s, bp_host) {
        var needs_bp = typeof bp_host === 'object';
        var G = new Graph(needs_bp);

        var l = 0;
        var am = null;
        do {
            am = G[this.hfn[l]](G.add(G.mul(this.W[l], s), this.b[l]));
            ++l;
        } while (l < this.hfn.length);

        var outmat = G.add(G.mul(this.W[l], am), this.b[l]);

        if (needs_bp)
            bp_host.G = G; // back this up. Kind of hacky isn't it

        return outmat;
    },
};

// Transformer definitions
var Graph = function(needs_backprop) {
    if(typeof needs_backprop === 'undefined') { needs_backprop = true; }
    this.needs_backprop = needs_backprop;

    // this will store a list of functions that perform backprop,
    // in their forward pass order. So in backprop we will go
    // backwards and evoke each one
    this.backprop = [];
};

Graph.prototype = {
    backward: function() {
        for(var i=this.backprop.length-1;i>=0;i--) {
            this.backprop[i](); // tick!
        }
    },
    rowPluck: function(m, ix) {
        // pluck a row of m with index ix and return it as col vector
        assert(ix >= 0 && ix < m.n);
        var d = m.d;
        var out = new Mat(d, 1);
        for(var i=0,n=d;i<n;i++){ out.w[i] = m.w[d * ix + i]; } // copy over the data

        if(this.needs_backprop) {
            var backward = function() {
                for(var i=0,n=d;i<n;i++){ m.dw[d * ix + i] += out.dw[i]; }
            }
            this.backprop.push(backward);
        }
        return out;
    },
    tanh: function(m) {
        // tanh nonlinearity
        var out = new Mat(m.n, m.d);
        var n = m.w.length;
        for(var i=0;i<n;i++) {
            out.w[i] = Math.tanh(m.w[i]);
        }

        if(this.needs_backprop) {
            var backward = function() {
                for(var i=0;i<n;i++) {
                    // grad for z = tanh(x) is (1 - z^2)
                    var mwi = out.w[i];
                    m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
                }
            }
            this.backprop.push(backward);
        }
        return out;
    },
    sigmoid: function(m) {
        // sigmoid nonlinearity
        var out = new Mat(m.n, m.d);
        var n = m.w.length;
        for(var i=0;i<n;i++) {
            out.w[i] = 1.0/(1+Math.exp(-m.w[i]));
        }

        if(this.needs_backprop) {
            var backward = function() {
                for(var i=0;i<n;i++) {
                    // grad for z = tanh(x) is (1 - z^2)
                    var mwi = out.w[i];
                    m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
                }
            }
            this.backprop.push(backward);
        }
        return out;
    },
    relu: function(m) {
        var out = new Mat(m.n, m.d);
        var n = m.w.length;
        for(var i=0;i<n;i++) {
            out.w[i] = Math.max(0, m.w[i]); // relu
        }
        if(this.needs_backprop) {
            var backward = function() {
                for(var i=0;i<n;i++) {
                    m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
                }
            }
            this.backprop.push(backward);
        }
        return out;
    },
    mul: function(m1, m2) {
        // multiply matrices m1 * m2
        assert(m1.d === m2.n, 'matmul dimensions misaligned');

        var n = m1.n;
        var d = m2.d;
        var out = new Mat(n,d);
        for(var i=0;i<m1.n;i++) { // loop over rows of m1
            for(var j=0;j<m2.d;j++) { // loop over cols of m2
                var dot = 0.0;
                for(var k=0;k<m1.d;k++) { // dot product loop
                    dot += m1.w[m1.d*i+k] * m2.w[m2.d*k+j];
                }
                out.w[d*i+j] = dot;
            }
        }

        if(this.needs_backprop) {
            var backward = function() {
                for(var i=0;i<m1.n;i++) { // loop over rows of m1
                    for(var j=0;j<m2.d;j++) { // loop over cols of m2
                        for(var k=0;k<m1.d;k++) { // dot product loop
                            var b = out.dw[d*i+j];
                            m1.dw[m1.d*i+k] += m2.w[m2.d*k+j] * b;
                            m2.dw[m2.d*k+j] += m1.w[m1.d*i+k] * b;
                        }
                    }
                }
            }
            this.backprop.push(backward);
        }
        return out;
    },
    add: function(m1, m2) {
        assert(m1.w.length === m2.w.length);

        var out = new Mat(m1.n, m1.d);
        for(var i=0,n=m1.w.length;i<n;i++) {
            out.w[i] = m1.w[i] + m2.w[i];
        }
        if(this.needs_backprop) {
            var backward = function() {
                for(var i=0,n=m1.w.length;i<n;i++) {
                    m1.dw[i] += out.dw[i];
                    m2.dw[i] += out.dw[i];
                }
            }
            this.backprop.push(backward);
        }
        return out;
    },
    dot: function(m1, m2) {
        // m1 m2 are both column vectors
        assert(m1.w.length === m2.w.length);
        var out = new Mat(1,1);
        var dot = 0.0;
        for(var i=0,n=m1.w.length;i<n;i++) {
            dot += m1.w[i] * m2.w[i];
        }
        out.w[0] = dot;
        if(this.needs_backprop) {
            var backward = function() {
                for(var i=0,n=m1.w.length;i<n;i++) {
                    m1.dw[i] += m2.w[i] * out.dw[0];
                    m2.dw[i] += m1.w[i] * out.dw[0];
                }
            }
            this.backprop.push(backward);
        }
        return out;
    },
    eltmul: function(m1, m2) {
        assert(m1.w.length === m2.w.length);

        var out = new Mat(m1.n, m1.d);
        for(var i=0,n=m1.w.length;i<n;i++) {
            out.w[i] = m1.w[i] * m2.w[i];
        }
        if(this.needs_backprop) {
            var backward = function() {
                for(var i=0,n=m1.w.length;i<n;i++) {
                    m1.dw[i] += m2.w[i] * out.dw[i];
                    m2.dw[i] += m1.w[i] * out.dw[i];
                }
            }
            this.backprop.push(backward);
        }
        return out;
    },
};

function DQN (params) {
    var that = new ReinforceLearning ( DQN, params);
    return that;
};

DQN.prototype.construct = function (params) {
    this.env = params.env;

    this.gamma = getopt(params, 'gamma', 0.75); // future reward discount factor
    this.epsilon = getopt(params, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(params, 'alpha', 0.01); // value function learning rate

    this.experience_add_every = getopt(params, 'experience_add_every', 25); // number of time steps before we add another experience to replay memory
    this.experience_size = getopt(params, 'experience_size', 5000); // size of experience replay
    this.learning_steps_per_iteration = getopt(params, 'learning_steps_per_iteration', 10);
    this.tderror_clamp = getopt(params, 'tderror_clamp', 1.0);

    this.converged = getopt(params, 'converged', false);

    this.num_hidden_units =  getopt(params, 'num_hidden_units', 100);
    this.nh = this.num_hidden_units;
    this.ns = getopt(params, 'ns', this.env.getNumStates());
    this.na = getopt(params, 'na', this.env.getMaxNumActions());
    this.nh = getopt(params, 'hiddens', [{n:this.num_hidden_units, fn:'tanh'}]);

    this.net = new Net(this.ns, this.na, this.nh);

    this.reset();

};

DQN.prototype.reset = function (force) {

    this.exp = []; // experience
    this.expi = 0; // where to insert

    this.t = 0;

    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;

    this.tderror = 0; // for visualization only...

    if (this.converged && !force) return;

    this.nh = [{n:this.num_hidden_units, fn:'tanh'}];
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();

    this.net = new Net(this.ns, this.na, this.nh);

};

DQN.prototype.actTrain = function(slist) {
    // convert to a Mat column vector
    var s = new Mat(this.ns, 1);
    s.setFrom(slist);

    // epsilon greedy policy
    if(Math.random() < this.epsilon) {
        var a = randi(0, this.na);
    } else {
        // greedy wrt Q function
        var amat = this.net.forward(s);
        var a = maxi(amat.w); // returns index of argmax action
    }

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;

    return a;
};

DQN.prototype.act = function(slist) {
    // convert to a Mat column vector
    var s = new Mat(this.ns, 1);
    s.setFrom(slist);

    // greedy wrt Q function
    var amat = this.net.forward(s);
    var a = maxi(amat.w); // returns index of argmax action

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;

    return a;
};

DQN.prototype.train = function(r1) {
    // perform an update on Q function
    if(!(this.r0 == null) && this.alpha > 0) {

        // learn from this tuple to get a sense of how "surprising" it is to the agent
        var tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1);
        this.tderror = tderror; // a measure of surprise

        // decide if we should keep this experience in the replay
        if(this.t % this.experience_add_every === 0) {
            this.exp[this.expi] = [this.s0, this.a0, this.r0, this.s1, this.a1];
            this.expi += 1;
            if(this.expi > this.experience_size) { this.expi = 0; } // roll over when we run out
        }
        this.t += 1;

        // sample some additional experience from replay memory and learn from it
        for(var k=0;k<this.learning_steps_per_iteration;k++) {
            var ri = randi(0, this.exp.length); // todo: priority sweeps?
            var e = this.exp[ri];
            this.learnFromTuple(e[0], e[1], e[2], e[3], e[4])
        }
    }
    this.r0 = r1; // store for next update
};

DQN.prototype.learnFromTuple = function(s0, a0, r0, s1, a1) {
    // want: Q(s,a) = r + gamma * max_a' Q(s',a')

    // compute the target Q value
    var tmat = this.net.forward(s1);
    var qmax = r0 + this.gamma * tmat.w[maxi(tmat.w)];

    // now predict
    var pred = this.net.forward(s0, this);

    var tderror = pred.w[a0] - qmax;
    var clamp = this.tderror_clamp;
    if(Math.abs(tderror) > clamp) {  // huber loss to robustify
        if(tderror > clamp) tderror = clamp;
        if(tderror < -clamp) tderror = -clamp;
    }
    pred.dw[a0] = tderror;
    this.G.backward(); // compute gradients on net params

    // update net
    this.net.update(this.alpha);
    return tderror;
};



