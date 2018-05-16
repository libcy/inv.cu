#pragma once

class NLCGOptimizer : public Optimizer {
protected:
    size_t nlcg_type;
    float nlcg_thresh;

    float pollak() {
        p_calc(p_new, 1, g_new, -1, g_old);
        float num = p_dot(g_new, p_new);
        float den = p_dot(g_old, g_old);
        return num / den;
    };
    float fletcher() {
        float num = p_dot(g_new, g_new);
        float den = p_dot(g_old, g_old);
        return num / den;
    };

    float calcStep(size_t step_count, float step_max, int &status) {
        return bracket(step_count, step_max, status);
    };

    int computeDirection() {
        inv_count++;
        if (inv_count == 1) {
            p_calc(p_new, -1, g_new);
            return 0;
        }
        else if(inv_cycle && inv_cycle < inv_count) {
            std::cout << "  restarting NLCG... [periodic restart]" << std::endl;
            return -1;
        }
        else {
            float beta;
            switch (nlcg_type) {
                case 0: beta = fletcher(); break;
                case 1: beta = pollak(); break;
                default: beta = 0;
            }
            p_calc(p_new, -1, g_new, beta, p_old);
            if(p_angle(g_new, g_old) < nlcg_thresh){
                std::cout << "  restarting NLCG... [loss of conjugacy]" << std::endl;
                return -1;
            }
            if(p_dot(p_new, g_new) > 0){
                std::cout << "  restarting NLCG... [not a descent direction]" << std::endl;
                return -1;
            }
            return 1;
        }
    };
public:
    void init(Config *config, Solver *solver, Misfit *misfit) {
        Optimizer::init(config, solver, misfit);
        nlcg_type = config->i["nlcg_type"];
        nlcg_thresh = config->f["nlcg_thresh"];
    };
};
