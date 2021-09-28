#ifndef DART_BIOMECH_MARKERFITTER_HPP_
#define DART_BIOMECH_MARKERFITTER_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

struct MarkerFitResult
{
  bool success;

  Eigen::VectorXs groupScales;
  std::vector<Eigen::VectorXs> poses;
  std::map<std::string, Eigen::Vector3s> markerErrors;

  Eigen::VectorXs rawMarkerOffsets;

  MarkerFitResult();
};

/**
 * This is the high level object that handles fitting skeletons to mocap data.
 *
 * It's supposed to take labeled point trajectories, and a known "marker set"
 * (where markers are, roughly, attached to the body), and use that to figure
 * out the body scales and the marker offsets (marker positions are never
 * perfect) that allow the best IK fit of the data.
 */
class MarkerFitter
{
public:
  MarkerFitter(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      dynamics::MarkerMap markers);

  /// This solves an optimization problem, trying to get the Skeleton to match
  /// the markers as closely as possible.
  std::shared_ptr<MarkerFitResult> optimize(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations);

  /// This lets us pick a subset of the marker observations, to cap the size of
  /// the optimization problem.
  static std::vector<std::map<std::string, Eigen::Vector3s>> pickSubset(
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      int maxSize);

  /// Internally all the markers are concatenated together, so each index has a
  /// name.
  std::string getMarkerNameAtIndex(int index);

  /// This method will set `skeleton` to the configuration given by the vectors
  /// of jointPositions and groupScales. It will also compute and return the
  /// list of markers given by markerDiffs.
  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
  setConfiguration(
      std::shared_ptr<dynamics::Skeleton>& skeleton,
      Eigen::VectorXs jointPositions,
      Eigen::VectorXs groupScales,
      Eigen::VectorXs markerDiffs);

  /// This computes a vector of concatenated differences between where markers
  /// are and where the observed markers are. Unobserved markers are assumed to
  /// have a difference of zero.
  Eigen::VectorXs getMarkerError(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the overall objective term for the MarkerFitter for a single
  /// timestep. The MarkerFitter is trying to do a bilevel optimization to
  /// minimize this term.
  s_t computeLoss(Eigen::VectorXs markerError);

  /// During random-restarts on IK, when we find solutions below this loss we'll
  /// stop doing restarts early, to speed up the process.
  void setInitialIKSatisfactoryLoss(s_t loss);

  /// This sets the maximum number of restarts allowed for the initial IK solver
  void setInitialIKMaxRestarts(int restarts);

  /// Sets the maximum that we'll allow markers to move from their original
  /// position, in meters
  void setMaxMarkerOffset(s_t offset);

  /// Sets the maximum number of iterations for IPOPT
  void setIterationLimit(int limit);

  //////////////////////////////////////////////////////////////////////////
  // First order gradients
  //////////////////////////////////////////////////////////////////////////

  /// This gets the gradient of the objective wrt the joint positions
  Eigen::VectorXs getLossGradientWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs markerError);

  /// This gets the gradient of the objective wrt the joint positions
  Eigen::VectorXs finiteDifferenceLossGradientWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the gradient of the objective wrt the group scales
  Eigen::VectorXs getLossGradientWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs markerError);

  /// This gets the gradient of the objective wrt the group scales
  Eigen::VectorXs finiteDifferenceLossGradientWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the gradient of the objective wrt the marker offsets
  Eigen::VectorXs getLossGradientWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs markerError);

  /// This gets the gradient of the objective wrt the marker offsets
  Eigen::VectorXs finiteDifferenceLossGradientWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  //////////////////////////////////////////////////////////////////////////
  // Jacobians of the gradient wrt joints (for bilevel optimization)
  //////////////////////////////////////////////////////////////////////////

  /// Get the marker indices that are not visible
  std::vector<int> getSparsityMap(
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the joints
  Eigen::MatrixXs getMarkerErrorJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the joints
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the joint positions
  Eigen::MatrixXs getLossGradientWrtJointsJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the joint positions
  Eigen::MatrixXs finiteDifferenceLossGradientWrtJointsJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the group scales
  Eigen::MatrixXs getMarkerErrorJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the group scales
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the group scales
  Eigen::MatrixXs getLossGradientWrtJointsJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the group scales
  Eigen::MatrixXs finiteDifferenceLossGradientWrtJointsJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the marker offsets
  Eigen::MatrixXs getMarkerErrorJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the marker offsets
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the marker offsets
  Eigen::MatrixXs getLossGradientWrtJointsJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the marker offsets
  Eigen::MatrixXs finiteDifferenceLossGradientWrtJointsJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  friend class BilevelFitProblem;

protected:
  std::map<std::string, int> mMarkerIndices;
  std::vector<std::string> mMarkerNames;

  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>> mMarkers;

  std::shared_ptr<dynamics::Skeleton> mSkeletonBallJoints;
  std::vector<std::pair<const dynamics::BodyNode*, Eigen::Vector3s>>
      mMarkersBallJoints;

  s_t mInitialIKSatisfactoryLoss;
  int mInitialIKMaxRestarts;
  s_t mMaxMarkerOffset;

  // These are IPOPT settings
  double mTolerance;
  int mIterationLimit;
  int mLBFGSHistoryLength;
  bool mCheckDerivatives;
  int mPrintFrequency;
  bool mSilenceOutput;
  bool mDisableLinesearch;
};

/*
 * Reminder: IPOPT will want to free this object when it's done with
 * optimization. This is responsible for actually transcribing the problem into
 * a format IPOpt can work with.
 */
class BilevelFitProblem : public Ipopt::TNLP
{
public:
  /// This creates a problem object. We take as arguments:
  /// @param skeleton the skeleton we're going to use to scale + fit the data
  /// @param markerSet the marker set we're using, with default offsets from the
  /// skeleton
  /// @param markerObservations a list of timesteps, where each timestep
  /// observes some subset of the markers at some points in 3D space.
  BilevelFitProblem(
      MarkerFitter* fitter,
      const std::vector<std::map<std::string, Eigen::Vector3s>>&
          markerObservations,
      std::shared_ptr<MarkerFitResult>& outResult);

  ~BilevelFitProblem();

  int getProblemSize();

  /// This gets a decent initial guess for the problem. We can guess scaling and
  /// joint positions from the first marker observation, and then use that
  /// scaling to get joint positions for all the other entries. This initially
  /// satisfies the constraint that we remain at optimal positional IK
  /// throughout optimization.
  Eigen::VectorXs getInitialization();

  /// This evaluates our loss function given a concatenated vector of all the
  /// problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
  s_t getLoss(Eigen::VectorXs x);

  /// This evaluates our gradient of loss given a concatenated vector of all
  /// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
  Eigen::VectorXs getGradient(Eigen::VectorXs x);

  /// This evaluates our gradient of loss given a concatenated vector of all
  /// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
  Eigen::VectorXs finiteDifferenceGradient(Eigen::VectorXs x);

  /// This evaluates our constraint vector given a concatenated vector of all
  /// the problem state: [groupSizes, markerOffsets, q_0, ..., q_N]
  Eigen::VectorXs getConstraints(Eigen::VectorXs x);

  /// This evaluates the Jacobian of our constraint vector wrt x given a
  /// concatenated vector of all the problem state: [groupSizes, markerOffsets,
  /// q_0, ..., q_N]
  Eigen::MatrixXs getConstraintsJacobian(Eigen::VectorXs x);

  /// This evaluates the Jacobian of our constraint vector wrt x given a
  /// concatenated vector of all the problem state: [groupSizes, markerOffsets,
  /// q_0, ..., q_N]
  Eigen::MatrixXs finiteDifferenceConstraintsJacobian(Eigen::VectorXs x);

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

protected:
  MarkerFitter* mFitter;
  std::vector<std::vector<std::pair<int, Eigen::Vector3s>>> mMarkerObservations;
  Eigen::VectorXs mObservationWeights;
  std::shared_ptr<MarkerFitResult>& mOutResult;
  bool mInitializationCached;
  Eigen::VectorXs mCachedInitialization;
};

} // namespace biomechanics

} // namespace dart

#endif