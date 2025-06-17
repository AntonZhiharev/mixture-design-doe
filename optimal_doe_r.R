# Optimal Design of Experiments (DOE) for n factors and m responses
# R implementation using AlgDesign and other packages

# Install required packages if not already installed
required_packages <- c("AlgDesign", "rsm", "DoE.base", "ggplot2", "gridExtra", "dplyr")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

#' Generate Optimal DOE Design
#' 
#' @param n_factors Number of factors
#' @param n_runs Number of experimental runs
#' @param factor_ranges List of ranges for each factor (min, max)
#' @param model_formula Model formula (default: quadratic)
#' @param criterion Optimality criterion ("D" or "I")
#' @param n_candidates Number of candidate points for selection
#' @return List containing design matrix and evaluation metrics
generate_optimal_doe <- function(n_factors, n_runs, factor_ranges = NULL, 
                                model_formula = NULL, criterion = "D", 
                                n_candidates = 1000) {
  
  # Set default factor ranges if not provided
  if (is.null(factor_ranges)) {
    factor_ranges <- replicate(n_factors, c(-1, 1), simplify = FALSE)
  }
  
  # Validate inputs
  if (length(factor_ranges) != n_factors) {
    stop("Number of factor ranges must match number of factors")
  }
  
  # Create factor names
  factor_names <- paste0("X", 1:n_factors)
  
  # Set default model formula (quadratic)
  if (is.null(model_formula)) {
    if (n_factors == 1) {
      model_formula <- ~ X1 + I(X1^2)
    } else if (n_factors == 2) {
      model_formula <- ~ X1 + X2 + I(X1^2) + I(X2^2) + X1:X2
    } else if (n_factors == 3) {
      model_formula <- ~ X1 + X2 + X3 + I(X1^2) + I(X2^2) + I(X3^2) + 
                       X1:X2 + X1:X3 + X2:X3
    } else {
      # For higher dimensions, create linear + interactions
      linear_terms <- paste(factor_names, collapse = " + ")
      quadratic_terms <- paste0("I(", factor_names, "^2)", collapse = " + ")
      model_formula <- as.formula(paste("~", linear_terms, "+", quadratic_terms))
    }
  }
  
  # Generate candidate set
  candidate_set <- data.frame(matrix(runif(n_candidates * n_factors), 
                                   ncol = n_factors))
  names(candidate_set) <- factor_names
  
  # Scale to factor ranges
  for (i in 1:n_factors) {
    min_val <- factor_ranges[[i]][1]
    max_val <- factor_ranges[[i]][2]
    candidate_set[, i] <- candidate_set[, i] * (max_val - min_val) + min_val
  }
  
  # Generate optimal design
  if (criterion == "D") {
    design_result <- optFederov(model_formula, candidate_set, nTrials = n_runs,
                               criterion = "D", evaluateI = TRUE)
  } else if (criterion == "I") {
    design_result <- optFederov(model_formula, candidate_set, nTrials = n_runs,
                               criterion = "I", evaluateI = TRUE)
  } else {
    stop("Criterion must be 'D' or 'I'")
  }
  
  # Extract design matrix
  design_matrix <- design_result$design[, factor_names]
  
  # Calculate additional metrics
  X <- as.matrix(design_matrix)
  model_matrix <- model.matrix(model_formula, data = design_matrix)
  
  # Calculate condition number
  XtX <- t(model_matrix) %*% model_matrix
  condition_number <- kappa(XtX)
  
  # Calculate variance inflation factors if possible
  vif_values <- NA
  try({
    if (ncol(model_matrix) > 1) {
      vif_values <- diag(solve(cor(model_matrix[, -1])))  # Exclude intercept
    }
  }, silent = TRUE)
  
  return(list(
    design = design_matrix,
    design_matrix = X,
    model_matrix = model_matrix,
    d_efficiency = design_result$Deff,
    i_efficiency = design_result$Ieff,
    condition_number = condition_number,
    vif = vif_values,
    n_runs = n_runs,
    n_factors = n_factors,
    criterion = criterion,
    factor_ranges = factor_ranges,
    model_formula = model_formula
  ))
}

#' Evaluate Design Quality
#' 
#' @param design Design matrix or data frame
#' @param model_formula Model formula for evaluation
#' @return List of design quality metrics
evaluate_design <- function(design, model_formula) {
  
  if (is.matrix(design)) {
    design <- as.data.frame(design)
    names(design) <- paste0("X", 1:ncol(design))
  }
  
  # Create model matrix
  model_matrix <- model.matrix(model_formula, data = design)
  
  # Calculate metrics
  XtX <- t(model_matrix) %*% model_matrix
  
  # D-efficiency
  d_eff <- det(XtX)^(1/ncol(model_matrix)) / nrow(design)
  
  # Condition number
  condition_number <- kappa(XtX)
  
  # A-efficiency (trace criterion)
  a_eff <- sum(diag(solve(XtX))) / ncol(model_matrix)
  
  return(list(
    d_efficiency = d_eff,
    condition_number = condition_number,
    a_efficiency = a_eff,
    n_runs = nrow(design),
    n_parameters = ncol(model_matrix)
  ))
}

#' Generate Multiple Response Analysis
#' 
#' @param designs List of design matrices
#' @param response_functions List of response functions
#' @param design_names Names for the designs
#' @return Data frame with response analysis
multiple_response_analysis <- function(designs, response_functions, design_names) {
  
  results <- data.frame()
  
  for (i in seq_along(designs)) {
    design <- designs[[i]]
    name <- design_names[i]
    
    row_data <- data.frame(
      Design = name,
      n_runs = nrow(design)
    )
    
    # Calculate responses for each function
    for (j in seq_along(response_functions)) {
      response_func <- response_functions[[j]]
      responses <- response_func(as.matrix(design))
      
      row_data[[paste0("Response_", j, "_mean")]] <- mean(responses)
      row_data[[paste0("Response_", j, "_sd")]] <- sd(responses)
      row_data[[paste0("Response_", j, "_range")]] <- diff(range(responses))
    }
    
    results <- rbind(results, row_data)
  }
  
  return(results)
}

#' Plot 2D Design
#' 
#' @param design Design matrix (2 factors only)
#' @param title Plot title
#' @param factor_names Names for the factors
plot_design_2d <- function(design, title = "Experimental Design", 
                          factor_names = c("Factor 1", "Factor 2")) {
  
  if (ncol(design) != 2) {
    stop("2D plotting only available for 2 factors")
  }
  
  design_df <- as.data.frame(design)
  names(design_df) <- factor_names
  design_df$Run <- 1:nrow(design_df)
  
  p <- ggplot(design_df, aes_string(x = factor_names[1], y = factor_names[2])) +
    geom_point(size = 4, color = "red", alpha = 0.7) +
    geom_text(aes(label = Run), vjust = -0.5, hjust = 0.5, size = 3) +
    labs(title = title, x = factor_names[1], y = factor_names[2]) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  print(p)
  return(p)
}

#' Compare Multiple Designs
#' 
#' @param designs List of design matrices
#' @param design_names Names for the designs
#' @param model_formula Model formula for evaluation
#' @return Data frame comparing design properties
compare_designs <- function(designs, design_names, model_formula) {
  
  comparison <- data.frame()
  
  for (i in seq_along(designs)) {
    design <- designs[[i]]
    name <- design_names[i]
    
    # Convert to data frame if matrix
    if (is.matrix(design)) {
      design_df <- as.data.frame(design)
      names(design_df) <- paste0("X", 1:ncol(design))
    } else {
      design_df <- design
    }
    
    # Evaluate design
    metrics <- evaluate_design(design_df, model_formula)
    
    row_data <- data.frame(
      Design = name,
      n_runs = metrics$n_runs,
      n_parameters = metrics$n_parameters,
      D_efficiency = metrics$d_efficiency,
      A_efficiency = metrics$a_efficiency,
      Condition_number = metrics$condition_number
    )
    
    comparison <- rbind(comparison, row_data)
  }
  
  return(comparison)
}

# ===== EXAMPLE USAGE AND DEMONSTRATION =====

# Set seed for reproducibility
set.seed(42)

cat("=== Optimal DOE Generator Demo (R) ===\n\n")

# Example 1: 2-factor D-optimal design
cat("1. Two-factor D-optimal design:\n")
n_factors <- 2
factor_ranges <- list(c(-2, 2), c(-1, 3))

d_optimal_result <- generate_optimal_doe(
  n_factors = n_factors,
  n_runs = 12,
  factor_ranges = factor_ranges,
  criterion = "D"
)

cat("D-optimal design (12 runs):\n")
cat("D-efficiency:", round(d_optimal_result$d_efficiency, 4), "\n")
cat("I-efficiency:", round(d_optimal_result$i_efficiency, 4), "\n")
cat("Condition number:", round(d_optimal_result$condition_number, 2), "\n")
cat("\nDesign points:\n")
print(round(d_optimal_result$design, 3))

# Example 2: 2-factor I-optimal design
cat("\n2. Two-factor I-optimal design:\n")
i_optimal_result <- generate_optimal_doe(
  n_factors = n_factors,
  n_runs = 12,
  factor_ranges = factor_ranges,
  criterion = "I"
)

cat("I-optimal design (12 runs):\n")
cat("D-efficiency:", round(i_optimal_result$d_efficiency, 4), "\n")
cat("I-efficiency:", round(i_optimal_result$i_efficiency, 4), "\n")
cat("Condition number:", round(i_optimal_result$condition_number, 2), "\n")
cat("\nDesign points:\n")
print(round(i_optimal_result$design, 3))

# Example 3: Three-factor design
cat("\n3. Three-factor D-optimal design:\n")
three_factor_result <- generate_optimal_doe(
  n_factors = 3,
  n_runs = 20,
  factor_ranges = list(c(-1, 1), c(-1, 1), c(-1, 1)),
  criterion = "D"
)

cat("3-factor D-optimal design (20 runs):\n")
cat("D-efficiency:", round(three_factor_result$d_efficiency, 4), "\n")
cat("I-efficiency:", round(three_factor_result$i_efficiency, 4), "\n")
cat("Condition number:", round(three_factor_result$condition_number, 2), "\n")

# Example 4: Multiple response analysis
cat("\n4. Multiple response analysis:\n")

# Define example response functions
response1 <- function(X) {
  # Linear response: y = 2*x1 + 3*x2 + noise
  return(2 * X[, 1] + 3 * X[, 2] + rnorm(nrow(X), 0, 0.1))
}

response2 <- function(X) {
  # Quadratic response: y = x1^2 + x2^2 + x1*x2 + noise
  return(X[, 1]^2 + X[, 2]^2 + X[, 1] * X[, 2] + rnorm(nrow(X), 0, 0.1))
}

response3 <- function(X) {
  # Interaction response: y = x1*x2 + 0.5*x1 + noise
  return(X[, 1] * X[, 2] + 0.5 * X[, 1] + rnorm(nrow(X), 0, 0.1))
}

# Compare designs for multiple responses
designs <- list(d_optimal_result$design, i_optimal_result$design)
design_names <- c("D-optimal", "I-optimal")
response_functions <- list(response1, response2, response3)

set.seed(42)  # For reproducible noise
comparison_df <- multiple_response_analysis(designs, response_functions, design_names)
cat("\nResponse comparison between designs:\n")
print(round(comparison_df, 4))

# Example 5: Design comparison
cat("\n5. Design property comparison:\n")
model_formula <- ~ X1 + X2 + I(X1^2) + I(X2^2) + X1:X2
design_comparison <- compare_designs(designs, design_names, model_formula)
cat("\nDesign comparison:\n")
print(round(design_comparison, 4))

# Example 6: Plotting (if ggplot2 is available)
cat("\n6. Generating plots...\n")
tryCatch({
  p1 <- plot_design_2d(d_optimal_result$design, "D-optimal Design")
  p2 <- plot_design_2d(i_optimal_result$design, "I-optimal Design")
  
  # Save plots
  ggsave("d_optimal_design.png", p1, width = 8, height = 6)
  ggsave("i_optimal_design.png", p2, width = 8, height = 6)
  cat("Plots saved as d_optimal_design.png and i_optimal_design.png\n")
}, error = function(e) {
  cat("Plotting requires ggplot2 and display capability\n")
})

# Example 7: Advanced - Custom model formula
cat("\n7. Custom model formula example:\n")
custom_formula <- ~ X1 + X2 + I(X1^3) + I(X2^3) + X1:X2
custom_design <- generate_optimal_doe(
  n_factors = 2,
  n_runs = 15,
  factor_ranges = list(c(-1, 1), c(-1, 1)),
  model_formula = custom_formula,
  criterion = "D"
)

cat("Custom cubic model design:\n")
cat("D-efficiency:", round(custom_design$d_efficiency, 4), "\n")
cat("I-efficiency:", round(custom_design$i_efficiency, 4), "\n")

cat("\n=== Demo Complete ===\n")

# Additional utility functions for advanced users

#' Generate Fractional Factorial Base Design
#' 
#' @param n_factors Number of factors
#' @param resolution Design resolution (III, IV, V)
#' @return Fractional factorial design
generate_fractional_factorial <- function(n_factors, resolution = "V") {
  
  if (n_factors <= 5) {
    # Full factorial for small designs
    design <- expand.grid(replicate(n_factors, c(-1, 1), simplify = FALSE))
  } else {
    # Use fractional factorial
    if (resolution == "III") {
      design <- FrF2(nruns = 2^(n_factors-1), nfactors = n_factors, resolution = 3)
    } else if (resolution == "IV") {
      design <- FrF2(nruns = 2^(n_factors-1), nfactors = n_factors, resolution = 4)
    } else {
      design <- FrF2(nruns = 2^(n_factors-1), nfactors = n_factors, resolution = 5)
    }
  }
  
  names(design) <- paste0("X", 1:n_factors)
  return(as.data.frame(design))
}

#' Augment Design with Center Points
#' 
#' @param design Base design matrix
#' @param n_center Number of center points to add
#' @return Augmented design with center points
add_center_points <- function(design, n_center = 3) {
  
  # Calculate center point (mean of factor ranges)
  center_point <- apply(design, 2, function(x) mean(range(x)))
  
  # Create center points
  center_points <- matrix(rep(center_point, n_center), nrow = n_center, byrow = TRUE)
  colnames(center_points) <- colnames(design)
  
  # Combine with original design
  augmented_design <- rbind(as.matrix(design), center_points)
  return(as.data.frame(augmented_design))
}

cat("\nAdditional utility functions loaded:\n")
cat("- generate_fractional_factorial()\n")
cat("- add_center_points()\n")
cat("- evaluate_design()\n")
cat("- compare_designs()\n")
