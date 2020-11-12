#ifndef DART_NEURAL_IPOPT_SHOT_WRAPPER_HPP_
#define DART_NEURAL_IPOPT_SHOT_WRAPPER_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/trajectory/AbstractShot.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {

class OptimizationRecord;

/*
 * IPOPT wants to own the trajectories it's trying to optimize, so we need a way
 * to create a buffer that's possible for IPOPT to own without freeing the
 * underlying trajectory when it's done.
 */
class IPOptShotWrapper : public Ipopt::TNLP
{
public:
  IPOptShotWrapper(
      AbstractShot* wrapped,
      std::shared_ptr<OptimizationRecord> record,
      bool recoverBest = true,
      bool recordFullDebugInfo = false,
      bool printIterations = false,
      bool recordIterations = true);

  /// Destructor
  ~IPOptShotWrapper();

  //------------------------- Ipopt::TNLP --------------------------------------
  /// \brief Method to return some info about the nlp
  bool get_nlp_info(
      Ipopt::Index& n,
      Ipopt::Index& m,
      Ipopt::Index& nnz_jac_g,
      Ipopt::Index& nnz_h_lag,
      Ipopt::TNLP::IndexStyleEnum& index_style) override;

  /// \brief Method to return the bounds for my problem
  bool get_bounds_info(
      Ipopt::Index n,
      Ipopt::Number* x_l,
      Ipopt::Number* x_u,
      Ipopt::Index m,
      Ipopt::Number* g_l,
      Ipopt::Number* g_u) override;

  /// \brief Method to return the starting point for the algorithm
  bool get_starting_point(
      Ipopt::Index n,
      bool init_x,
      Ipopt::Number* x,
      bool init_z,
      Ipopt::Number* z_L,
      Ipopt::Number* z_U,
      Ipopt::Index m,
      bool init_lambda,
      Ipopt::Number* lambda) override;

  /// \brief Method to return the objective value
  bool eval_f(
      Ipopt::Index _n,
      const Ipopt::Number* _x,
      bool _new_x,
      Ipopt::Number& _obj_value) override;

  /// \brief Method to return the gradient of the objective
  bool eval_grad_f(
      Ipopt::Index _n,
      const Ipopt::Number* _x,
      bool _new_x,
      Ipopt::Number* _grad_f) override;

  /// \brief Method to return the constraint residuals
  bool eval_g(
      Ipopt::Index _n,
      const Ipopt::Number* _x,
      bool _new_x,
      Ipopt::Index _m,
      Ipopt::Number* _g) override;

  /// \brief Method to return:
  ///        1) The structure of the jacobian (if "values" is nullptr)
  ///        2) The values of the jacobian (if "values" is not nullptr)
  bool eval_jac_g(
      Ipopt::Index _n,
      const Ipopt::Number* _x,
      bool _new_x,
      Ipopt::Index _m,
      Ipopt::Index _nele_jac,
      Ipopt::Index* _iRow,
      Ipopt::Index* _jCol,
      Ipopt::Number* _values) override;

  /// \brief Method to return:
  ///        1) The structure of the hessian of the lagrangian (if "values" is
  ///           nullptr)
  ///        2) The values of the hessian of the lagrangian (if "values" is not
  ///           nullptr)
  bool eval_h(
      Ipopt::Index _n,
      const Ipopt::Number* _x,
      bool _new_x,
      Ipopt::Number _obj_factor,
      Ipopt::Index _m,
      const Ipopt::Number* _lambda,
      bool _new_lambda,
      Ipopt::Index _nele_hess,
      Ipopt::Index* _iRow,
      Ipopt::Index* _jCol,
      Ipopt::Number* _values) override;

  /// \brief This method is called when the algorithm is complete so the TNLP
  ///        can store/write the solution
  void finalize_solution(
      Ipopt::SolverReturn _status,
      Ipopt::Index _n,
      const Ipopt::Number* _x,
      const Ipopt::Number* _z_L,
      const Ipopt::Number* _z_U,
      Ipopt::Index _m,
      const Ipopt::Number* _g,
      const Ipopt::Number* _lambda,
      Ipopt::Number _obj_value,
      const Ipopt::IpoptData* _ip_data,
      Ipopt::IpoptCalculatedQuantities* _ip_cq) override;

  bool intermediate_callback(
      Ipopt::AlgorithmMode mode,
      Ipopt::Index iter,
      Ipopt::Number obj_value,
      Ipopt::Number inf_pr,
      Ipopt::Number inf_du,
      Ipopt::Number mu,
      Ipopt::Number d_norm,
      Ipopt::Number regularization_size,
      Ipopt::Number alpha_du,
      Ipopt::Number alpha_pr,
      Ipopt::Index ls_trials,
      const Ipopt::IpoptData* ip_data,
      Ipopt::IpoptCalculatedQuantities* ip_cq) override;

  /// This gets called when we're about to repoptimize, to let us reset values.
  void prep_for_reoptimize();

  /// This records a single call of eval_f(). If this returns false, then we
  /// need to terminate this call to eval_f().
  bool can_eval_f(bool new_x);

  /// This records a single call of eval_grad_f(). If this returns false, then
  /// we need to terminate this call to eval_grad_f().
  bool can_eval_grad_f(bool new_x);

  /// This records a single call of eval_g(). If this returns false, then
  /// we need to terminate this call to eval_g().
  bool can_eval_g(bool new_x);

  /// This records a single call of eval_jac_g(). If this returns false, then
  /// we need to terminate this call to eval_jac_g().
  bool can_eval_jac_g(bool new_x);

  /// This is a central method that evaluates if we can continue the
  /// optimization
  bool can_continue();

  /// This resets the stored data from this iteration
  void reset_iteration();

private:
  AbstractShot* mWrapped;
  std::shared_ptr<OptimizationRecord> mRecord;
  bool mRecoverBest;
  bool mRecordFullDebugInfo;
  bool mRecordIterations;
  int mBestIter;
  double mBestFeasibleObjectiveValue;
  Eigen::VectorXd mBestFeasibleState;
  bool mPrintIterations;
  long mLastTimestep;

  int mNewXs;
  int mFCalls;
  int mGradFCalls;
  int mGCalls;
  int mJacGCalls;

  Eigen::VectorXd mSaved_zU;
  Eigen::VectorXd mSaved_zL;
  Eigen::VectorXd mSaved_lambda;
};

} // namespace trajectory
} // namespace dart

#endif