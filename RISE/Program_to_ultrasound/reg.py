import SimpleITK as sitk
import numpy as np

#Defining constants for the registration
LEARING_RATE = 2.0
MIN_STEP = 1e-4
NUMBER_OF_ITERATIONS = 100
GRADIENT_MAGNITUDE_TOLERANCE = 1e-8

#I wanted to use multi-threading but I think it is not necessary for this task
def register(fixed_image, moving_image):

    #Convert numpy arrays to SimpleITK images
    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)

    #Inıtıal alignment of two images
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    #Similarity metric settings
    registration_method.SetMetricAsMeanSquares()

    #Interpolator settings
    registration_method.SetInterpolator(sitk.sitkLinear)

    #Optimizer settings
    #Maybe not optimal for our case but time is limited and I think it is a good start
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=LEARING_RATE, minStep=MIN_STEP, numberOfIterations=NUMBER_OF_ITERATIONS, gradientMagnitudeTolerance=GRADIENT_MAGNITUDE_TOLERANCE)

    #Execute the registration
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed_image, moving_image)

    #Apply the final transformation to the moving image
    resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    #Example usage of the function

fixed_image = np.load("fixed_image.npy")
moving_image = np.load("moving_image.npy")
registered_image = register(fixed_image, moving_image)