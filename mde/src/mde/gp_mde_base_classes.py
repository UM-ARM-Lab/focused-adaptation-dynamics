import gpytorch
import torch
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.models import ExactGP
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch.models.gp import GP

from moonshine.gpytorch_tools import custom_combine_batches, _fix_env2, mutate_dict_to_cuda


class DictExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("ExactGP can only handle Gaussian likelihoods")
        super(ExactGP, self).__init__()
        if train_inputs is not None:
            self.train_inputs = train_inputs  # tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None
        self.likelihood = likelihood
        self.prediction_strategy = None

    def _apply(self, fn):
        if self.train_inputs is not None:
            if isinstance(self.train_inputs, dict):
                for key_name in self.train_inputs.keys():
                    if isinstance(self.train_inputs[key_name], torch.Tensor):
                        self.train_inputs[key_name] = fn(self.train_inputs[key_name])
            else:
                self.train_inputs = tuple([fn(train_input) for train_input in self.train_inputs])
            self.train_targets = fn(self.train_targets)
        return super(ExactGP, self)._apply(fn)

    def __call__(self, *args, **kwargs):
        if isinstance(args[0], dict):
            inputs = args[0]
        else:
            train_inputs = []
            inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]
        train_inputs = self.train_inputs
        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if False and settings.debug.on():
                if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    raise RuntimeError("You must train on the training inputs!")
            if isinstance(inputs, dict):
                res = GP.__call__(self, inputs, **kwargs)
                # res = super().__call__(inputs, **kwargs)
            else:
                res = super().__call__(*inputs, **kwargs)
            return res
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output
            # Posterior mode
        else:
            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                if isinstance(train_inputs, dict):
                    # train_output = super().__call__(train_inputs, **kwargs)
                    train_output = GP.__call__(self, train_inputs, **kwargs)
                else:
                    train_output = ExactGP.__call__(self, *train_inputs, **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )
            # Concatenate the input to the training input
            full_inputs = []
            full_inputs = custom_combine_batches([self.train_inputs, inputs])
            full_output = super(ExactGP, self).__call__(full_inputs, **kwargs)
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])
            # Make the prediction
            with settings._use_eval_tolerance():
                predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)


class DeepExactGP(DictExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        train_x["env"] = _fix_env2(train_x["env"])
        super(DeepExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean().cuda()
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepExactGPDetachedTrainingData(DictExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, final_hidden_dim, grid_size,
                 features_to_gp_input, input_data_to_features):

        train_x_features_detached = input_data_to_features(train_x).detach()
        super(DeepExactGPDetachedTrainingData, self).__init__(train_x_features_detached, train_y, likelihood)
        self.input_data_to_features = input_data_to_features
        self.features_to_gp_input = features_to_gp_input
        gpytorch.models.ExactGP.__init__(self, train_x_features_detached, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        if isinstance(x, dict):
            projected_x = self.feature_extractor(x)
        else:  # assuming is training data...kind of hacky
            projected_x = self.features_to_gp_input(x).cuda()
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs):
        if isinstance(args[0], dict):
            inputs = args[0]
            for arg in args:
                mutate_dict_to_cuda(arg)
        else:
            train_inputs = []
            inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]
        # train_inputs = tuple([self.features_to_gp_input(train_input) for train_input in self.train_inputs])
        train_inputs = self.train_inputs
        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if isinstance(inputs, dict):
                res = GP.__call__(self, inputs, **kwargs)
            else:
                res = super().__call__(*inputs, **kwargs)
            return res
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output
            # Posterior mode
        else:
            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                if True or isinstance(train_inputs, dict):
                    # train_output = super().__call__(train_inputs, **kwargs)
                    train_output = GP.__call__(self, train_inputs[0], **kwargs)
                else:
                    train_output = ExactGP.__call__(self, *train_inputs, **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )
            # Concatenate the input to the training input
            transformed_inputs = self.input_data_to_features(inputs)
            full_inputs = torch.cat([train_inputs[0], transformed_inputs], dim=-2)
            # Need to do this cleverly
            full_output = super(ExactGP, self).__call__(full_inputs, **kwargs)
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])
            # Make the prediction
            with settings._use_eval_tolerance():
                predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)
