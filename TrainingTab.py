import datetime
import json
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import tflite_model_maker
from tflite_model_maker import object_detector

from TrainingToolTip import ToolTip


class TrainingTab:
    def __init__(self, parent, log_queue):
        self.parent = parent
        self.log_queue = log_queue
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.training_thread = None
        self.stop_flag = threading.Event()
        self.current_model = None
        self.training_history = []
        self.tooltips_enabled = tk.BooleanVar(value=True)
        self.tooltip_widgets = []
        self.setup_ui()

    def setup_ui(self):
        # Main frame directly in parent - no canvas/scrollbar
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Tooltip toggle at the top
        tooltip_frame = ttk.Frame(main_frame)
        tooltip_frame.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(tooltip_frame, text="Show parameter tooltips",
                        variable=self.tooltips_enabled,
                        command=self.toggle_tooltips).pack(side=tk.LEFT)

        # Section 1: Dataset Configuration
        ttk.Label(main_frame, text="Dataset Configuration",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        dataset_frame = ttk.LabelFrame(main_frame, text="Training Data Directories", padding="10")
        dataset_frame.pack(fill="x", pady=(0, 20))

        # Training directory
        train_frame = ttk.Frame(dataset_frame)
        train_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(train_frame, text="Training Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.train_dir_var = tk.StringVar()
        ttk.Entry(train_frame, textvariable=self.train_dir_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(train_frame, text="Browse",
                   command=lambda: self.browse_directory(self.train_dir_var)).grid(row=0, column=2, padx=(5, 0))
        train_frame.columnconfigure(1, weight=1)

        # Validation directory
        valid_frame = ttk.Frame(dataset_frame)
        valid_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(valid_frame, text="Validation Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.valid_dir_var = tk.StringVar()
        ttk.Entry(valid_frame, textvariable=self.valid_dir_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(valid_frame, text="Browse",
                   command=lambda: self.browse_directory(self.valid_dir_var)).grid(row=0, column=2, padx=(5, 0))
        valid_frame.columnconfigure(1, weight=1)

        # Test directory
        test_frame = ttk.Frame(dataset_frame)
        test_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(test_frame, text="Test Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.test_dir_var = tk.StringVar()
        ttk.Entry(test_frame, textvariable=self.test_dir_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(test_frame, text="Browse",
                   command=lambda: self.browse_directory(self.test_dir_var)).grid(row=0, column=2, padx=(5, 0))
        test_frame.columnconfigure(1, weight=1)

        # Analyze data button
        ttk.Button(dataset_frame, text="Analyze Dataset",
                   command=self.analyze_dataset).pack(pady=(10, 0))

        # Section 2: Model Configuration
        ttk.Label(main_frame, text="Model Configuration",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(20, 10))

        model_frame = ttk.LabelFrame(main_frame, text="Model Settings", padding="10")
        model_frame.pack(fill="x", pady=(0, 20))

        # Label configuration
        label_frame = ttk.Frame(model_frame)
        label_frame.pack(fill="x", pady=(0, 10))
        label_label = ttk.Label(label_frame, text="Object Label:")
        label_label.pack(side=tk.LEFT, padx=(0, 10))
        self.label_var = tk.StringVar(value="kettlebell")
        ttk.Entry(label_frame, textvariable=self.label_var, width=20).pack(side=tk.LEFT)
        self.add_tooltip(label_label, "The class name for your object (e.g., 'kettlebell', 'person', 'car')")

        # Model architecture
        arch_frame = ttk.Frame(model_frame)
        arch_frame.pack(fill="x", pady=(0, 10))
        arch_label = ttk.Label(arch_frame, text="Model Architecture:")
        arch_label.pack(side=tk.LEFT, padx=(0, 10))
        self.architecture_var = tk.StringVar(value="efficientdet_lite0")
        ttk.Label(arch_frame, text="EfficientDet-Lite0 (320x320 input)").pack(side=tk.LEFT)
        self.add_tooltip(arch_label, "EfficientDet-Lite0 is optimized for mobile/edge devices with 320x320 input size")

        # Backbone selection
        backbone_frame = ttk.Frame(model_frame)
        backbone_frame.pack(fill="x", pady=(0, 10))
        backbone_label = ttk.Label(backbone_frame, text="Backbone:")
        backbone_label.pack(side=tk.LEFT, padx=(0, 10))
        self.backbone_var = tk.StringVar(value="efficientnet-b0")
        backbone_combo = ttk.Combobox(backbone_frame, textvariable=self.backbone_var,
                                      values=["efficientnet-b0", "efficientnet-b1",
                                              "efficientnet-b2", "efficientnet-b3"],
                                      width=30)
        backbone_combo.pack(side=tk.LEFT)
        self.add_tooltip(backbone_label,
                         "Feature extractor network. Higher numbers (b0→b3) are larger/more accurate but slower")

        # Training mode
        mode_frame = ttk.Frame(model_frame)
        mode_frame.pack(fill="x", pady=(0, 10))
        self.train_whole_model_var = tk.BooleanVar(value=True)
        whole_model_cb = ttk.Checkbutton(mode_frame, text="Train whole model (not just head)",
                                         variable=self.train_whole_model_var)
        whole_model_cb.pack(side=tk.LEFT)
        self.add_tooltip(whole_model_cb,
                         "Train entire network (slower but more accurate) vs just the detection head (faster)")

        # Section 3: Training Hyperparameters
        ttk.Label(main_frame, text="Training Hyperparameters",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(20, 10))

        # Create notebook for hyperparameter categories
        hyper_notebook = ttk.Notebook(main_frame)
        hyper_notebook.pack(fill="x", pady=(0, 20))

        # Basic hyperparameters tab
        basic_frame = ttk.Frame(hyper_notebook)
        hyper_notebook.add(basic_frame, text="Basic")

        basic_params = ttk.Frame(basic_frame, padding="10")
        basic_params.pack(fill="x")

        # Epochs
        epochs_frame = ttk.Frame(basic_params)
        epochs_frame.pack(fill="x", pady=(0, 5))
        epochs_label = ttk.Label(epochs_frame, text="Epochs:")
        epochs_label.pack(side=tk.LEFT, padx=(0, 10))
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(epochs_frame, textvariable=self.epochs_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(epochs_label, "Number of complete passes through the training dataset")

        # Batch size
        batch_frame = ttk.Frame(basic_params)
        batch_frame.pack(fill="x", pady=(0, 5))
        batch_label = ttk.Label(batch_frame, text="Batch Size:")
        batch_label.pack(side=tk.LEFT, padx=(0, 10))
        self.batch_size_var = tk.StringVar(value="8")
        ttk.Entry(batch_frame, textvariable=self.batch_size_var, width=10).pack(side=tk.LEFT)
        ttk.Label(batch_frame, text="(lower if GPU memory issues)").pack(side=tk.LEFT, padx=(10, 0))
        self.add_tooltip(batch_label, "Number of images processed together. Lower values use less memory")

        # Learning rate
        lr_frame = ttk.Frame(basic_params)
        lr_frame.pack(fill="x", pady=(0, 5))
        lr_label = ttk.Label(lr_frame, text="Learning Rate:")
        lr_label.pack(side=tk.LEFT, padx=(0, 10))
        self.learning_rate_var = tk.StringVar(value="0.008")
        ttk.Entry(lr_frame, textvariable=self.learning_rate_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(lr_label, "Step size for weight updates. Lower = more stable, Higher = faster training")

        # LR Warmup
        warmup_frame = ttk.Frame(basic_params)
        warmup_frame.pack(fill="x", pady=(0, 5))
        warmup_label = ttk.Label(warmup_frame, text="LR Warmup Init:")
        warmup_label.pack(side=tk.LEFT, padx=(0, 10))
        self.lr_warmup_init_var = tk.StringVar(value="0.001")
        ttk.Entry(warmup_frame, textvariable=self.lr_warmup_init_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(warmup_label, "Initial learning rate for warmup period. Gradually increases to main LR")

        # Advanced hyperparameters tab
        advanced_frame = ttk.Frame(hyper_notebook)
        hyper_notebook.add(advanced_frame, text="Advanced")

        advanced_params = ttk.Frame(advanced_frame, padding="10")
        advanced_params.pack(fill="x")

        # Create two columns for advanced parameters
        adv_left = ttk.Frame(advanced_params)
        adv_left.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 20))

        adv_right = ttk.Frame(advanced_params)
        adv_right.pack(side=tk.LEFT, fill="both", expand=True)

        # Left column - Optimization parameters
        ttk.Label(adv_left, text="Optimization", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        # Optimizer
        self.optimizer_var = tk.StringVar(value="sgd")
        opt_frame = ttk.Frame(adv_left)
        opt_frame.pack(fill="x", pady=(0, 5))
        opt_label = ttk.Label(opt_frame, text="Optimizer:")
        opt_label.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Combobox(opt_frame, textvariable=self.optimizer_var,
                     values=["sgd", "adam"], width=10).pack(side=tk.LEFT)
        self.add_tooltip(opt_label, "SGD: stable but slower, Adam: faster convergence but may be less stable")

        # Momentum
        self.momentum_var = tk.StringVar(value="0.9")
        mom_frame = ttk.Frame(adv_left)
        mom_frame.pack(fill="x", pady=(0, 5))
        mom_label = ttk.Label(mom_frame, text="Momentum:")
        mom_label.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(mom_frame, textvariable=self.momentum_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(mom_label, "Helps accelerate SGD in relevant direction (0.9 is standard)")

        # Weight decay
        self.weight_decay_var = tk.StringVar(value="4e-5")
        wd_frame = ttk.Frame(adv_left)
        wd_frame.pack(fill="x", pady=(0, 5))
        wd_label = ttk.Label(wd_frame, text="Weight Decay:")
        wd_label.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(wd_frame, textvariable=self.weight_decay_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(wd_label, "L2 regularization to prevent overfitting (1e-5 to 1e-4 typical)")

        # Gradient clipping
        self.gradient_clip_var = tk.StringVar(value="10.0")
        gc_frame = ttk.Frame(adv_left)
        gc_frame.pack(fill="x", pady=(0, 5))
        gc_label = ttk.Label(gc_frame, text="Gradient Clip:")
        gc_label.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(gc_frame, textvariable=self.gradient_clip_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(gc_label, "Maximum gradient norm to prevent exploding gradients")

        # Label smoothing
        self.label_smoothing_var = tk.StringVar(value="0.1")
        ls_frame = ttk.Frame(adv_left)
        ls_frame.pack(fill="x", pady=(0, 5))
        ls_label = ttk.Label(ls_frame, text="Label Smoothing:")
        ls_label.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(ls_frame, textvariable=self.label_smoothing_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(ls_label, "Softens hard labels to prevent overconfidence (0.1 is good default)")

        # Right column - Augmentation parameters
        ttk.Label(adv_right, text="Augmentation", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        # AutoAugment
        self.autoaugment_var = tk.StringVar(value="v2")
        aug_frame = ttk.Frame(adv_right)
        aug_frame.pack(fill="x", pady=(0, 5))
        aug_label = ttk.Label(aug_frame, text="AutoA ugment:")
        aug_label.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Combobox(aug_frame, textvariable=self.autoaugment_var,
                     values=["v0", "v1", "v2", "v3", "None"], width=10).pack(side=tk.LEFT)
        self.add_tooltip(aug_label, "Automatic augmentation policy. v2/v3 are stronger, None disables")

        # Jitter min
        self.jitter_min_var = tk.StringVar(value="0.1")
        jmin_frame = ttk.Frame(adv_right)
        jmin_frame.pack(fill="x", pady=(0, 5))
        jmin_label = ttk.Label(jmin_frame, text="Jitter Min:")
        jmin_label.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(jmin_frame, textvariable=self.jitter_min_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(jmin_label, "Minimum scale factor for random resizing (0.1 = 10% of original)")

        # Jitter max
        self.jitter_max_var = tk.StringVar(value="2.0")
        jmax_frame = ttk.Frame(adv_right)
        jmax_frame.pack(fill="x", pady=(0, 5))
        jmax_label = ttk.Label(jmax_frame, text="Jitter Max:")
        jmax_label.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(jmax_frame, textvariable=self.jitter_max_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(jmax_label, "Maximum scale factor for random resizing (2.0 = 200% of original)")

        # Random horizontal flip
        self.input_rand_hflip_var = tk.BooleanVar(value=True)
        hflip_cb = ttk.Checkbutton(adv_right, text="Random Horizontal Flip",
                                   variable=self.input_rand_hflip_var)
        hflip_cb.pack(anchor=tk.W)
        self.add_tooltip(hflip_cb, "Randomly flip images horizontally for augmentation")

        # Grid mask
        self.grid_mask_var = tk.BooleanVar(value=False)
        grid_cb = ttk.Checkbutton(adv_right, text="Grid Mask Augmentation",
                                  variable=self.grid_mask_var)
        grid_cb.pack(anchor=tk.W)
        self.add_tooltip(grid_cb, "Apply grid mask augmentation (can improve robustness)")

        # Loss Configuration tab
        loss_frame = ttk.Frame(hyper_notebook)
        hyper_notebook.add(loss_frame, text="Loss Config")

        loss_params = ttk.Frame(loss_frame, padding="10")
        loss_params.pack(fill="x")

        # Focal loss alpha
        self.alpha_var = tk.StringVar(value="0.25")
        alpha_frame = ttk.Frame(loss_params)
        alpha_frame.pack(fill="x", pady=(0, 5))
        alpha_label = ttk.Label(alpha_frame, text="Focal Loss Alpha:")
        alpha_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(alpha_frame, textvariable=self.alpha_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(alpha_label, "Balances positive/negative examples in focal loss")

        # Focal loss gamma
        self.gamma_var = tk.StringVar(value="1.5")
        gamma_frame = ttk.Frame(loss_params)
        gamma_frame.pack(fill="x", pady=(0, 5))
        gamma_label = ttk.Label(gamma_frame, text="Focal Loss Gamma:")
        gamma_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(gamma_frame, textvariable=self.gamma_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(gamma_label, "Focusing parameter for hard examples (1.5-2.0 typical)")

        # Box loss weight
        self.box_loss_weight_var = tk.StringVar(value="50.0")
        box_weight_frame = ttk.Frame(loss_params)
        box_weight_frame.pack(fill="x", pady=(0, 5))
        box_weight_label = ttk.Label(box_weight_frame, text="Box Loss Weight:")
        box_weight_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(box_weight_frame, textvariable=self.box_loss_weight_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(box_weight_label, "Weight for bounding box regression loss")

        # IoU loss
        iou_frame = ttk.Frame(loss_params)
        iou_frame.pack(fill="x", pady=(0, 5))
        iou_label = ttk.Label(iou_frame, text="IoU Loss Type:")
        iou_label.pack(side=tk.LEFT, padx=(0, 10))
        self.iou_loss_type_var = tk.StringVar(value="None")
        ttk.Combobox(iou_frame, textvariable=self.iou_loss_type_var,
                     values=["None", "iou", "ciou", "diou", "giou"], width=15).pack(side=tk.LEFT)
        self.add_tooltip(iou_label, "Additional IoU-based loss. CIoU/DIoU often work best")

        # IoU loss weight
        self.iou_loss_weight_var = tk.StringVar(value="1.0")
        iou_weight_frame = ttk.Frame(loss_params)
        iou_weight_frame.pack(fill="x", pady=(0, 5))
        iou_weight_label = ttk.Label(iou_weight_frame, text="IoU Loss Weight:")
        iou_weight_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(iou_weight_frame, textvariable=self.iou_loss_weight_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(iou_weight_label, "Weight for IoU loss (if enabled)")

        # NMS Configuration tab
        nms_frame = ttk.Frame(hyper_notebook)
        hyper_notebook.add(nms_frame, text="NMS Config")

        nms_params = ttk.Frame(nms_frame, padding="10")
        nms_params.pack(fill="x")

        # NMS method
        self.nms_method_var = tk.StringVar(value="gaussian")
        nms_method_frame = ttk.Frame(nms_params)
        nms_method_frame.pack(fill="x", pady=(0, 5))
        nms_method_label = ttk.Label(nms_method_frame, text="NMS Method:")
        nms_method_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Combobox(nms_method_frame, textvariable=self.nms_method_var,
                     values=["gaussian", "hard"], width=15).pack(side=tk.LEFT)
        self.add_tooltip(nms_method_label, "Gaussian: soft suppression, Hard: traditional NMS")

        # Score threshold
        self.score_thresh_var = tk.StringVar(value="0.0")
        score_frame = ttk.Frame(nms_params)
        score_frame.pack(fill="x", pady=(0, 5))
        score_label = ttk.Label(score_frame, text="Score Threshold:")
        score_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(score_frame, textvariable=self.score_thresh_var, width=10).pack(side=tk.LEFT)
        ttk.Label(score_frame, text="(0.0-1.0, higher = fewer detections)").pack(side=tk.LEFT, padx=(10, 0))
        self.add_tooltip(score_label, "Minimum confidence score for detections")

        # IoU threshold
        self.iou_thresh_var = tk.StringVar(value="0.5")
        iou_thresh_frame = ttk.Frame(nms_params)
        iou_thresh_frame.pack(fill="x", pady=(0, 5))
        iou_thresh_label = ttk.Label(iou_thresh_frame, text="IoU Threshold:")
        iou_thresh_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(iou_thresh_frame, textvariable=self.iou_thresh_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(iou_thresh_label, "Maximum IoU between boxes before suppression")

        # Max output size
        self.max_output_size_var = tk.StringVar(value="100")
        max_out_frame = ttk.Frame(nms_params)
        max_out_frame.pack(fill="x", pady=(0, 5))
        max_out_label = ttk.Label(max_out_frame, text="Max Detections:")
        max_out_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(max_out_frame, textvariable=self.max_output_size_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(max_out_label, "Maximum number of detections per image")

        # Anchor Configuration tab
        anchor_frame = ttk.Frame(hyper_notebook)
        hyper_notebook.add(anchor_frame, text="Anchors")

        anchor_params = ttk.Frame(anchor_frame, padding="10")
        anchor_params.pack(fill="x")

        # Anchor scale
        self.anchor_scale_var = tk.StringVar(value="4.0")
        anchor_scale_frame = ttk.Frame(anchor_params)
        anchor_scale_frame.pack(fill="x", pady=(0, 5))
        anchor_scale_label = ttk.Label(anchor_scale_frame, text="Anchor Scale:")
        anchor_scale_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(anchor_scale_frame, textvariable=self.anchor_scale_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(anchor_scale_label, "Base anchor size multiplier (4.0 is standard)")

        # Aspect ratios
        aspect_frame = ttk.Frame(anchor_params)
        aspect_frame.pack(fill="x", pady=(0, 5))
        aspect_label = ttk.Label(aspect_frame, text="Aspect Ratios:")
        aspect_label.pack(side=tk.LEFT, padx=(0, 10))
        self.aspect_ratios_var = tk.StringVar(value="1.0,2.0,0.5")
        ttk.Entry(aspect_frame, textvariable=self.aspect_ratios_var, width=20).pack(side=tk.LEFT)
        ttk.Label(aspect_frame, text="(comma-separated)").pack(side=tk.LEFT, padx=(5, 0))
        self.add_tooltip(aspect_label, "Width/height ratios for anchor boxes (e.g., 1.0,2.0,0.5)")

        # Max instances
        self.max_instances_var = tk.StringVar(value="100")
        max_inst_frame = ttk.Frame(anchor_params)
        max_inst_frame.pack(fill="x", pady=(0, 5))
        max_inst_label = ttk.Label(max_inst_frame, text="Max Instances per Image:")
        max_inst_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(max_inst_frame, textvariable=self.max_instances_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(max_inst_label, "Maximum object instances to consider per image during training")

        # Learning Rate Schedule tab
        lr_frame = ttk.Frame(hyper_notebook)
        hyper_notebook.add(lr_frame, text="LR Schedule")

        lr_params = ttk.Frame(lr_frame, padding="10")
        lr_params.pack(fill="x")

        # LR decay method
        self.lr_decay_method_var = tk.StringVar(value="cosine")
        lr_method_frame = ttk.Frame(lr_params)
        lr_method_frame.pack(fill="x", pady=(0, 5))
        lr_method_label = ttk.Label(lr_method_frame, text="LR Decay Method:")
        lr_method_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Combobox(lr_method_frame, textvariable=self.lr_decay_method_var,
                     values=["cosine", "polynomial", "stepwise"], width=15).pack(side=tk.LEFT)
        self.add_tooltip(lr_method_label, "How learning rate decreases over time")

        # First LR drop
        self.first_lr_drop_var = tk.StringVar(value="0.7")
        first_drop_frame = ttk.Frame(lr_params)
        first_drop_frame.pack(fill="x", pady=(0, 5))
        first_drop_label = ttk.Label(first_drop_frame, text="First LR Drop (fraction):")
        first_drop_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(first_drop_frame, textvariable=self.first_lr_drop_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(first_drop_label, "When to first reduce LR (0.7 = 70% through training)")

        # Second LR drop
        self.second_lr_drop_var = tk.StringVar(value="0.9")
        second_drop_frame = ttk.Frame(lr_params)
        second_drop_frame.pack(fill="x", pady=(0, 5))
        second_drop_label = ttk.Label(second_drop_frame, text="Second LR Drop (fraction):")
        second_drop_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(second_drop_frame, textvariable=self.second_lr_drop_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(second_drop_label, "When to second reduce LR (0.9 = 90% through training)")

        # Moving average decay
        self.moving_average_decay_var = tk.StringVar(value="0.9998")
        ma_frame = ttk.Frame(lr_params)
        ma_frame.pack(fill="x", pady=(0, 5))
        ma_label = ttk.Label(ma_frame, text="Moving Average Decay:")
        ma_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(ma_frame, textvariable=self.moving_average_decay_var, width=10).pack(side=tk.LEFT)
        self.add_tooltip(ma_label, "Exponential moving average for model weights (improves stability)")

        # Preset configurations
        preset_frame = ttk.Frame(main_frame)
        preset_frame.pack(fill="x", pady=(0, 20))

        ttk.Label(preset_frame, text="Preset Configurations:").pack(side=tk.LEFT, padx=(0, 10))
        self.preset_var = tk.StringVar(value="custom")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                    values=["custom", "default", "high_accuracy", "fast_training", "mobile_optimized"],
                                    width=20)
        preset_combo.pack(side=tk.LEFT)
        preset_combo.bind('<<ComboboxSelected>>', self.load_preset)

        ttk.Button(preset_frame, text="Save Current as Preset",
                   command=self.save_preset).pack(side=tk.LEFT, padx=(10, 0))

        # Section 4: Training Controls
        ttk.Label(main_frame, text="Training Controls",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(20, 10))

        control_frame = ttk.LabelFrame(main_frame, text="Training Actions", padding="10")
        control_frame.pack(fill="x", pady=(0, 20))

        # Export directory
        export_frame = ttk.Frame(control_frame)
        export_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(export_frame, text="Export Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.export_dir_var = tk.StringVar(value="models")
        ttk.Entry(export_frame, textvariable=self.export_dir_var, width=40).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(export_frame, text="Browse",
                   command=lambda: self.browse_directory(self.export_dir_var)).grid(row=0, column=2, padx=(5, 0))
        export_frame.columnconfigure(1, weight=1)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=(10, 5))

        self.status_label = ttk.Label(control_frame, text="Ready to train")
        self.status_label.pack()

        # Training buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=(10, 0))

        self.train_btn = ttk.Button(button_frame, text="Start Training",
                                    command=self.start_training)
        self.train_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop Training",
                                   command=self.stop_training, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.evaluate_btn = ttk.Button(button_frame, text="Evaluate Model",
                                       command=self.evaluate_model, state='disabled')
        self.evaluate_btn.grid(row=0, column=2, padx=5)

        self.export_btn = ttk.Button(button_frame, text="Export Model",
                                     command=self.export_model, state='disabled')
        self.export_btn.grid(row=0, column=3, padx=5)

        # Training history display
        history_frame = ttk.LabelFrame(main_frame, text="Training History", padding="10")
        history_frame.pack(fill="both", expand=True, pady=(20, 0))

        self.history_text = scrolledtext.ScrolledText(history_frame, height=10, wrap=tk.WORD)
        self.history_text.pack(fill="both", expand=True)

    def add_tooltip(self, widget, text):
        """Add a tooltip to a widget"""
        tooltip = ToolTip(widget, text)
        self.tooltip_widgets.append((widget, tooltip))
        # Set initial state based on checkbox
        tooltip.enabled = self.tooltips_enabled.get()

    def toggle_tooltips(self):
        """Toggle all tooltips on/off"""
        enabled = self.tooltips_enabled.get()
        for widget, tooltip in self.tooltip_widgets:
            tooltip.enabled = enabled

    def load_preset(self, event=None):
        preset = self.preset_var.get()

        if preset == "default":
            self.epochs_var.set("50")
            self.batch_size_var.set("8")
            self.learning_rate_var.set("0.008")
            self.lr_warmup_init_var.set("0.001")
            self.optimizer_var.set("sgd")
            self.momentum_var.set("0.9")
            self.weight_decay_var.set("4e-5")
            self.gradient_clip_var.set("10.0")
            self.label_smoothing_var.set("0.1")
            self.autoaugment_var.set("v2")
            self.score_thresh_var.set("0.0")

        elif preset == "high_accuracy":
            self.epochs_var.set("100")
            self.batch_size_var.set("4")
            self.learning_rate_var.set("0.004")
            self.lr_warmup_init_var.set("0.0008")
            self.optimizer_var.set("sgd")
            self.momentum_var.set("0.9")
            self.weight_decay_var.set("5e-5")
            self.gradient_clip_var.set("10.0")
            self.label_smoothing_var.set("0.1")
            self.autoaugment_var.set("v3")
            self.jitter_min_var.set("0.1")
            self.jitter_max_var.set("2.0")
            self.grid_mask_var.set(True)

        elif preset == "fast_training":
            self.epochs_var.set("25")
            self.batch_size_var.set("16")
            self.learning_rate_var.set("0.01")
            self.lr_warmup_init_var.set("0.001")
            self.optimizer_var.set("adam")
            self.momentum_var.set("0.9")
            self.weight_decay_var.set("1e-5")
            self.gradient_clip_var.set("10.0")
            self.autoaugment_var.set("v1")
            self.grid_mask_var.set(False)

        elif preset == "mobile_optimized":
            self.epochs_var.set("50")
            self.batch_size_var.set("8")
            self.learning_rate_var.set("0.008")
            self.backbone_var.set("efficientnet-b0")
            self.train_whole_model_var.set(False)
            self.max_output_size_var.set("50")
            self.score_thresh_var.set("0.3")

    def save_preset(self):
        # Implementation for saving custom presets
        messagebox.showinfo("Info", "Preset saving functionality to be implemented")

    def _training_thread(self):
        try:
            self.log_queue.put("\n=== Starting Model Training ===")
            self._update_status("Loading datasets...")

            # Create label map
            label_map = {1: self.label_var.get()}

            # Load datasets
            train_data = object_detector.DataLoader.from_pascal_voc(
                images_dir=self.train_dir_var.get(),
                annotations_dir=self.train_dir_var.get(),
                label_map=label_map
            )

            valid_data = object_detector.DataLoader.from_pascal_voc(
                images_dir=self.valid_dir_var.get(),
                annotations_dir=self.valid_dir_var.get(),
                label_map=label_map
            )

            self.test_data = object_detector.DataLoader.from_pascal_voc(
                images_dir=self.test_dir_var.get(),
                annotations_dir=self.test_dir_var.get(),
                label_map=label_map
            )

            # Create model spec with all hyperparameters
            self._update_status("Creating model specification...")

            # Parse aspect ratios
            aspect_ratios = [float(x.strip()) for x in self.aspect_ratios_var.get().split(',')]

            # Calculate epoch-based LR drops
            epochs = int(self.epochs_var.get())
            first_drop_epoch = int(float(self.first_lr_drop_var.get()) * epochs)
            second_drop_epoch = int(float(self.second_lr_drop_var.get()) * epochs)

            hparams = {
                'backbone_name': self.backbone_var.get(),
                'image_size': 320,  # Fixed for lite0
                'num_classes': 1,  # Single object class
                'learning_rate': float(self.learning_rate_var.get()),
                'lr_warmup_init': float(self.lr_warmup_init_var.get()),
                'lr_warmup_epoch': 1.0,
                'first_lr_drop_epoch': first_drop_epoch,
                'second_lr_drop_epoch': second_drop_epoch,
                'lr_decay_method': self.lr_decay_method_var.get(),
                'moving_average_decay': float(self.moving_average_decay_var.get()),
                'optimizer': self.optimizer_var.get(),
                'momentum': float(self.momentum_var.get()),
                'weight_decay': float(self.weight_decay_var.get()),
                'gradient_clip_norm': float(self.gradient_clip_var.get()),
                'label_smoothing': float(self.label_smoothing_var.get()),
                'alpha': float(self.alpha_var.get()),
                'gamma': float(self.gamma_var.get()),
                'box_loss_weight': float(self.box_loss_weight_var.get()),
                'iou_loss_type': None if self.iou_loss_type_var.get() == "None" else self.iou_loss_type_var.get(),
                'iou_loss_weight': float(self.iou_loss_weight_var.get()),
                'anchor_scale': float(self.anchor_scale_var.get()),
                'aspect_ratios': aspect_ratios,
                'max_instances_per_image': int(self.max_instances_var.get()),
                'input_rand_hflip': self.input_rand_hflip_var.get(),
                'jitter_min': float(self.jitter_min_var.get()),
                'jitter_max': float(self.jitter_max_var.get()),
                'autoaugment_policy': None if self.autoaugment_var.get() == "None" else self.autoaugment_var.get(),
                'grid_mask': self.grid_mask_var.get(),
                'nms_configs': {
                    'method': self.nms_method_var.get(),
                    'iou_thresh': float(self.iou_thresh_var.get()),
                    'score_thresh': float(self.score_thresh_var.get()),
                    'sigma': 0.5 if self.nms_method_var.get() == 'gaussian' else None,
                    'max_output_size': int(self.max_output_size_var.get())
                },
                'tflite_max_detections': int(self.max_output_size_var.get()),
                'strategy': None,  # Will be set based on available hardware
                'num_epochs': epochs
            }

            model_spec = tflite_model_maker.object_detector.EfficientDetSpec(
                model_name='efficientdet-lite0',
                uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
                hparams=hparams
            )

            # Create and train model
            self._update_status("Creating model...")
            self.current_model = object_detector.create(
                train_data=train_data,
                model_spec=model_spec,
                batch_size=int(self.batch_size_var.get()),
                train_whole_model=self.train_whole_model_var.get(),
                validation_data=valid_data,
                epochs=epochs,
                do_train=True
            )

            if not self.stop_flag.is_set():
                self._update_status("Training completed!")
                self._update_progress(100)
                self.log_queue.put("\n✅ Training completed successfully!")

                # Enable evaluation and export
                self.parent.after(0, lambda: self.evaluate_btn.config(state='normal'))
                self.parent.after(0, lambda: self.export_btn.config(state='normal'))

        except Exception as e:
            self.log_queue.put(f"\n❌ Training error: {str(e)}")
            self._update_status(f"Error: {str(e)}")

        finally:
            self.parent.after(0, lambda: self.train_btn.config(state='normal'))
            self.parent.after(0, lambda: self.stop_btn.config(state='disabled'))

    def _save_training_config(self, filepath):
        config = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": {
                "train_dir": self.train_dir_var.get(),
                "valid_dir": self.valid_dir_var.get(),
                "test_dir": self.test_dir_var.get(),
                "label": self.label_var.get()
            },
            "model": {
                "architecture": self.architecture_var.get(),
                "backbone": self.backbone_var.get(),
                "train_whole_model": self.train_whole_model_var.get()
            },
            "hyperparameters": {
                "epochs": self.epochs_var.get(),
                "batch_size": self.batch_size_var.get(),
                "learning_rate": self.learning_rate_var.get(),
                "lr_warmup_init": self.lr_warmup_init_var.get(),
                "lr_decay_method": self.lr_decay_method_var.get(),
                "first_lr_drop": self.first_lr_drop_var.get(),
                "second_lr_drop": self.second_lr_drop_var.get(),
                "moving_average_decay": self.moving_average_decay_var.get(),
                "optimizer": self.optimizer_var.get(),
                "momentum": self.momentum_var.get(),
                "weight_decay": self.weight_decay_var.get(),
                "gradient_clip": self.gradient_clip_var.get(),
                "label_smoothing": self.label_smoothing_var.get(),
                "alpha": self.alpha_var.get(),
                "gamma": self.gamma_var.get(),
                "box_loss_weight": self.box_loss_weight_var.get(),
                "iou_loss_type": self.iou_loss_type_var.get(),
                "iou_loss_weight": self.iou_loss_weight_var.get(),
                "anchor_scale": self.anchor_scale_var.get(),
                "aspect_ratios": self.aspect_ratios_var.get(),
                "max_instances_per_image": self.max_instances_var.get(),
                "autoaugment_policy": self.autoaugment_var.get(),
                "jitter_min": self.jitter_min_var.get(),
                "jitter_max": self.jitter_max_var.get(),
                "input_rand_hflip": self.input_rand_hflip_var.get(),
                "grid_mask": self.grid_mask_var.get()
            },
            "nms_config": {
                "method": self.nms_method_var.get(),
                "score_thresh": self.score_thresh_var.get(),
                "iou_thresh": self.iou_thresh_var.get(),
                "max_output_size": self.max_output_size_var.get()
            },
            "training_history": self.training_history
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    # ... rest of the methods remain the same ...
    def browse_directory(self, var):
        directory = filedialog.askdirectory(
            title="Select Directory",
            initialdir=self.app_root
        )
        if directory:
            var.set(directory)

    def analyze_dataset(self):
        train_dir = self.train_dir_var.get()
        valid_dir = self.valid_dir_var.get()
        test_dir = self.test_dir_var.get()

        if not all([train_dir, valid_dir, test_dir]):
            messagebox.showwarning("Warning", "Please select all three directories.")
            return

        self.log_queue.put("\n=== Dataset Analysis ===")

        try:
            # Create label map
            label_map = {1: self.label_var.get()}

            # Load datasets
            train_data = object_detector.DataLoader.from_pascal_voc(
                images_dir=train_dir,
                annotations_dir=train_dir,
                label_map=label_map
            )

            valid_data = object_detector.DataLoader.from_pascal_voc(
                images_dir=valid_dir,
                annotations_dir=valid_dir,
                label_map=label_map
            )

            test_data = object_detector.DataLoader.from_pascal_voc(
                images_dir=test_dir,
                annotations_dir=test_dir,
                label_map=label_map
            )

            self.log_queue.put(f"Training images: {train_data.size}")
            self.log_queue.put(f"Validation images: {valid_data.size}")
            self.log_queue.put(f"Test images: {test_data.size}")
            self.log_queue.put(f"Label: {self.label_var.get()}")

            # Check for image size variations
            self.log_queue.put(
                "\nNote: EfficientDet-Lite0 will automatically resize all images to 320x320 during training.")
            self.log_queue.put("Your source images can be any size - higher resolution is better for augmentation.")

        except Exception as e:
            self.log_queue.put(f"Error analyzing dataset: {str(e)}")
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

    def start_training(self):
        # Validate inputs
        if not all([self.train_dir_var.get(), self.valid_dir_var.get(), self.test_dir_var.get()]):
            messagebox.showwarning("Warning", "Please select all dataset directories.")
            return

        if not self.export_dir_var.get():
            messagebox.showwarning("Warning", "Please select an export directory.")
            return

        # Disable buttons
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        # Clear history
        self.history_text.delete(1.0, tk.END)
        self.training_history = []

        # Start training thread
        self.stop_flag.clear()
        self.training_thread = threading.Thread(target=self._training_thread, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        self.stop_flag.set()
        self.log_queue.put("Stopping training...")

    def evaluate_model(self):
        if not self.current_model or not hasattr(self, 'test_data'):
            messagebox.showwarning("Warning", "No trained model available for evaluation.")
            return

        self.log_queue.put("\n=== Evaluating Model ===")

        try:
            batch_size = int(self.batch_size_var.get())
            metrics = self.current_model.evaluate(self.test_data, batch_size=batch_size)

            self.log_queue.put("Evaluation Results:")
            for key, value in metrics.items():
                self.log_queue.put(f"  {key}: {value}")

            # Add to history
            self._add_to_history(f"\nFinal Evaluation: AP={metrics.get('AP', 'N/A')}")

        except Exception as e:
            self.log_queue.put(f"Error during evaluation: {str(e)}")
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")

    def export_model(self):
        if not self.current_model:
            messagebox.showwarning("Warning", "No trained model available for export.")
            return

        export_dir = self.export_dir_var.get()
        if not export_dir:
            messagebox.showwarning("Warning", "Please specify an export directory.")
            return

        try:
            os.makedirs(export_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            architecture = self.architecture_var.get()
            label = self.label_var.get()

            if self.train_whole_model_var.get():
                filename = f"{architecture}_{label}_whole_{timestamp}.tflite"
            else:
                filename = f"{architecture}_{label}_head_{timestamp}.tflite"

            self.log_queue.put(f"\n=== Exporting Model ===")
            self.log_queue.put(f"Export directory: {export_dir}")
            self.log_queue.put(f"Filename: {filename}")

            # Export the model
            self.current_model.export(export_dir=export_dir, tflite_filename=filename)

            # Save training configuration
            config_file = os.path.join(export_dir, f"training_config_{timestamp}.json")
            self._save_training_config(config_file)

            self.log_queue.put(f"\n✅ Model exported successfully!")
            self.log_queue.put(f"Model file: {os.path.join(export_dir, filename)}")
            self.log_queue.put(f"Config file: {config_file}")

            messagebox.showinfo("Success", f"Model exported to:\n{os.path.join(export_dir, filename)}")

        except Exception as e:
            self.log_queue.put(f"Error during export: {str(e)}")
            messagebox.showerror("Error", f"Export failed: {str(e)}")

    def _update_status(self, message):
        self.parent.after(0, lambda: self.status_label.config(text=message))

    def _update_progress(self, value):
        self.parent.after(0, lambda: self.progress_var.set(value))

    def _add_to_history(self, message):
        self.parent.after(0, lambda: self.history_text.insert(tk.END, message + '\n'))
        self.parent.after(0, lambda: self.history_text.see(tk.END))
        self.training_history.append(message)
