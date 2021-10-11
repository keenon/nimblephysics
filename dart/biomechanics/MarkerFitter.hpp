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

class MarkerFitter;
struct MarkerFitterState;

struct MarkerFitResult
{
  bool success;

  Eigen::VectorXs groupScales;
  std::vector<Eigen::VectorXs> poses;
  Eigen::MatrixXs posesMatrix;
  std::map<std::string, Eigen::Vector3s> markerErrors;

  Eigen::VectorXs rawMarkerOffsets;

  MarkerFitResult();
};

/**
 * This gives us a legible version of what's going on inside the optimization
 * problem, so it's easier to unpack specific parts of the loss in useful ways.
 */
struct MarkerFitterState
{
  // The current state
  std::map<std::string, Eigen::Vector3s> bodyScales;
  std::map<std::string, Eigen::Vector3s> markerOffsets;
  std::vector<std::map<std::string, Eigen::Vector3s>> markerErrorsAtTimesteps;
  std::vector<Eigen::VectorXs> posesAtTimesteps;

  // The gradient of the current state, which is not always used, but can help
  // shuttling information back and forth from friendly PyTorch APIs.
  std::map<std::string, Eigen::Vector3s> bodyScalesGrad;
  std::map<std::string, Eigen::Vector3s> markerOffsetsGrad;
  std::vector<std::map<std::string, Eigen::Vector3s>>
      markerErrorsAtTimestepsGrad;
  std::vector<Eigen::VectorXs> posesAtTimestepsGrad;

  /// This unflattens an input vector, given some information about the problm
  MarkerFitterState(
      const Eigen::VectorXs& flat,
      std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
      MarkerFitter* fitter);

  /// This returns a single flat vector representing this whole problem state
  Eigen::VectorXs flattenState();

  /// This returns a single flat vector representing the gradient of this whole
  /// problem state
  Eigen::VectorXs flattenGradient();

protected:
  std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations;
  std::shared_ptr<dynamics::Skeleton> skeleton;
  std::vector<std::string> markerOrder;
  MarkerFitter* fitter;
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

  /// All markers are either "anatomical" or "tracking". Markers are presumed to
  /// be anamotical markers unless otherwise specified. Tracking markers are
  /// treated differently - they're not used in the initial scaling and fitting,
  /// and their initial positions are not trusted at all. Instead, during
  /// initialization, we guess their offset based on where the markers are
  /// observed to be.
  void setMarkerIsTracking(std::string marker, bool isTracking = true);

  /// This returns true if the given marker is "tracking", otherwise it's
  /// "anatomical"
  bool getMarkerIsTracking(std::string marker);

  /// This auto-labels any markers whose names end with '1', '2', or '3' as
  /// tracking markers, on the assumption that they're tracking triads.
  void setTriadsToTracking();

  /// Gets the total number of markers we've got in this Fitter
  int getNumMarkers();

  /// Internally all the markers are concatenated together, so each index has a
  /// name.
  std::string getMarkerNameAtIndex(int index);

  /// Internally all the markers are concatenated together, so each index has a
  /// name.
  int getMarkerIndex(std::string name);

  /// This method will set `skeleton` to the configuration given by the vectors
  /// of jointPositions and groupScales. It will also compute and return the
  /// list of markers given by markerDiffs.
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> setConfiguration(
      std::shared_ptr<dynamics::Skeleton>& skeleton,
      Eigen::VectorXs jointPositions,
      Eigen::VectorXs groupScales,
      Eigen::VectorXs markerDiffs);

  /// This computes a vector of concatenated differences between where markers
  /// are and where the observed markers are. Unobserved markers are assumed to
  /// have a difference of zero.
  Eigen::VectorXs getMarkerError(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the overall objective term for the MarkerFitter for a single
  /// timestep. The MarkerFitter is trying to do a bilevel optimization to
  /// minimize this term.
  s_t computeIKLoss(Eigen::VectorXs markerError);

  /// This returns the gradient for the simple IK loss term
  Eigen::VectorXs getIKLossGradWrtMarkerError(Eigen::VectorXs markerError);

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

  /// Sets the loss and gradient function
  void setCustomLossAndGrad(
      std::function<s_t(MarkerFitterState*)> customLossAndGrad);

  /// This adds a custom function as an equality constraint to the problem. The
  /// constraint has to equal 0.
  void addZeroConstraint(
      std::string name,
      std::function<s_t(MarkerFitterState*)> customConstraintAndGrad);

  /// This removes an equality constraint by name
  void removeZeroConstraint(std::string name);

  //////////////////////////////////////////////////////////////////////////
  // First order gradients
  //////////////////////////////////////////////////////////////////////////

  /// This gets the gradient of the objective wrt the joint positions
  Eigen::VectorXs getMarkerLossGradientWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs lossGradWrtMarkerError);

  /// This gets the gradient of the objective wrt the joint positions
  Eigen::VectorXs finiteDifferenceSquaredMarkerLossGradientWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the gradient of the objective wrt the group scales
  Eigen::VectorXs getMarkerLossGradientWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs lossGradWrtMarkerError);

  /// This gets the gradient of the objective wrt the group scales
  Eigen::VectorXs finiteDifferenceSquaredMarkerLossGradientWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the gradient of the objective wrt the marker offsets
  Eigen::VectorXs getMarkerLossGradientWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs lossGradWrtMarkerError);

  /// This gets the gradient of the objective wrt the marker offsets
  Eigen::VectorXs finiteDifferenceSquaredMarkerLossGradientWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  //////////////////////////////////////////////////////////////////////////
  // Jacobians of the gradient wrt joints (for bilevel optimization)
  //////////////////////////////////////////////////////////////////////////

  /// Get the marker indices that are not visible
  std::vector<int> getSparsityMap(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the joints
  Eigen::MatrixXs getMarkerErrorJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the joints
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the joint positions
  Eigen::MatrixXs getIKLossGradientWrtJointsJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the joint positions
  Eigen::MatrixXs finiteDifferenceIKLossGradientWrtJointsJacobianWrtJoints(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the group scales
  Eigen::MatrixXs getMarkerErrorJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the group scales
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the group scales
  Eigen::MatrixXs getIKLossGradientWrtJointsJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the group scales
  Eigen::MatrixXs finiteDifferenceIKLossGradientWrtJointsJacobianWrtGroupScales(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of the marker error wrt the marker offsets
  Eigen::MatrixXs getMarkerErrorJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of the marker error wrt the marker offsets
  Eigen::MatrixXs finiteDifferenceMarkerErrorJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the marker offsets
  Eigen::MatrixXs getIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const Eigen::VectorXs& markerError,
      const std::vector<int>& sparsityMap);

  /// This gets the jacobian of (the gradient of the objective wrt the joint
  /// positions) wrt the marker offsets
  Eigen::MatrixXs
  finiteDifferenceIKLossGradientWrtJointsJacobianWrtMarkerOffsets(
      const std::shared_ptr<dynamics::Skeleton>& skeleton,
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      const std::vector<std::pair<int, Eigen::Vector3s>>&
          visibleMarkerWorldPoses);

  friend class BilevelFitProblem;
  friend struct MarkerFitterState;

protected:
  std::map<std::string, int> mMarkerIndices;
  std::vector<std::string> mMarkerNames;
  std::vector<bool> mMarkerIsTracking;

  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> mMarkers;
  std::vector<dynamics::Joint*> mObservedJoints;
  dynamics::MarkerMap mMarkerMap;

  std::shared_ptr<dynamics::Skeleton> mSkeletonBallJoints;
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      mMarkersBallJoints;

  std::function<s_t(MarkerFitterState*)> mLossAndGrad;
  std::map<std::string, std::function<s_t(MarkerFitterState*)>>
      mZeroConstraints;

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
  std::vector<std::map<std::string, Eigen::Vector3s>> mMarkerMapObservations;
  std::vector<std::vector<std::pair<int, Eigen::Vector3s>>> mMarkerObservations;
  Eigen::VectorXs mObservationWeights;
  std::shared_ptr<MarkerFitResult>& mOutResult;
  bool mInitializationCached;
  Eigen::VectorXs mCachedInitialization;
};

} // namespace biomechanics

} // namespace dart

#endif