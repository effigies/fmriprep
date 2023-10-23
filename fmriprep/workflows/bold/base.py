# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Orchestrating the BOLD-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_wf
.. autofunction:: init_bold_fit_wf
.. autofunction:: init_bold_native_wf

"""
import os
import typing as ty

import nibabel as nb
import numpy as np
from nipype.interfaces import utility as niu
from nipype.interfaces.fsl import Split as FSLSplit
from nipype.pipeline import engine as pe
from niworkflows.utils.connections import listify, pop_file

from ... import config
from ...interfaces import DerivativesDataSink
from ...interfaces.reports import FunctionalSummary
from ...utils.meepi import combine_meepi_source

# BOLD workflows
from .apply import init_bold_volumetric_resample_wf
from .confounds import init_bold_confs_wf, init_carpetplot_wf
from .fit import init_bold_fit_wf, init_bold_native_wf, init_vol2surf_fit_wf
from .hmc import init_bold_hmc_wf
from .outputs import (
    init_ds_bold_native_wf,
    init_ds_registration_wf,
    init_ds_volumes_wf,
    init_func_derivatives_wf,
)
from .registration import init_bold_reg_wf, init_bold_t1_trans_wf
from .resampling import (
    init_bold_preproc_trans_wf,
    init_bold_std_trans_wf,
    init_bold_surf_wf,
)
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf, init_t2s_reporting_wf


def init_bold_wf(
    *,
    bold_series: ty.List[str],
    precomputed: dict = {},
    fieldmap_id: ty.Optional[str] = None,
    name: str = "bold_wf",
) -> pe.Workflow:
    """
    This workflow controls the functional preprocessing stages of *fMRIPrep*.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.tests import mock_config
            from fmriprep import config
            from fmriprep.workflows.bold.base import init_bold_wf
            with mock_config():
                bold_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
                wf = init_bold_wf(
                    bold_series=[str(bold_file)],
                )

    Parameters
    ----------
    bold_series
        List of paths to NIfTI files.
    precomputed
        Dictionary containing precomputed derivatives to reuse, if possible.
    fieldmap_id
        ID of the fieldmap to use to correct this BOLD series. If :obj:`None`,
        no correction will be applied.

    Inputs
    ------
    t1w_preproc
        Bias-corrected structural template image
    t1w_mask
        Mask of the skull-stripped template image
    t1w_dseg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    anat2std_xfm
        List of transform files, collated with templates
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    fsnative2t1w_xfm
        LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w
    fmap_id
        Unique identifiers to select fieldmap files
    fmap
        List of estimated fieldmaps (collated with fmap_id)
    fmap_ref
        List of fieldmap reference files (collated with fmap_id)
    fmap_coeff
        List of lists of spline coefficient files (collated with fmap_id)
    fmap_mask
        List of fieldmap masks (collated with fmap_id)
    sdc_method
        List of fieldmap correction method names (collated with fmap_id)

    See Also
    --------

    * :func:`~fmriprep.workflows.bold.fit.init_bold_fit_wf`
    * :func:`~fmriprep.workflows.bold.fit.init_bold_native_wf`
    * :func:`~fmriprep.workflows.bold.outputs.init_ds_bold_native_wf`
    * :func:`~fmriprep.workflows.bold.t2s.init_t2s_reporting_wf`

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    bold_file = bold_series[0]

    fmriprep_dir = config.execution.fmriprep_dir
    omp_nthreads = config.nipype.omp_nthreads
    all_metadata = [config.execution.layout.get_metadata(file) for file in bold_series]

    nvols, mem_gb = _create_mem_gb(bold_file)
    if nvols <= 5 - config.execution.sloppy:
        config.loggers.workflow.warning(
            f"Too short BOLD series (<= 5 timepoints). Skipping processing of <{bold_file}>."
        )
        return

    config.loggers.workflow.debug(
        "Creating bold processing workflow for <%s> (%.2f GB / %d TRs). "
        "Memory resampled/largemem=%.2f/%.2f GB.",
        bold_file,
        mem_gb["filesize"],
        nvols,
        mem_gb["resampled"],
        mem_gb["largemem"],
    )

    functional_cache = {}
    if config.execution.derivatives:
        from fmriprep.utils.bids import collect_derivatives, extract_entities

        entities = extract_entities(bold_series)

        for deriv_dir in config.execution.derivatives:
            functional_cache.update(
                collect_derivatives(
                    derivatives_dir=deriv_dir,
                    entities=entities,
                    fieldmap_id=fieldmap_id,
                )
            )

    workflow = Workflow(name=_get_wf_name(bold_file, "bold"))

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Anatomical coregistration
                "t1w_preproc",
                "t1w_mask",
                "t1w_dseg",
                # FreeSurfer outputs
                "subjects_dir",
                "subject_id",
                "fsnative2t1w_xfm",
                "white",
                "midthickness",
                "pial",
                "anat_ribbon",
                # Fieldmap registration
                "fmap",
                "fmap_ref",
                "fmap_coeff",
                "fmap_mask",
                "fmap_id",
                "sdc_method",
            ],
        ),
        name="inputnode",
    )

    #
    # Minimal workflow
    #

    bold_fit_wf = init_bold_fit_wf(
        bold_series=bold_series,
        precomputed=functional_cache,
        fieldmap_id=fieldmap_id,
        omp_nthreads=omp_nthreads,
    )

    workflow.connect([
        (inputnode, bold_fit_wf, [
            ('t1w_preproc', 'inputnode.t1w_preproc'),
            ('t1w_mask', 'inputnode.t1w_mask'),
            ('t1w_dseg', 'inputnode.t1w_dseg'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
            ('fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm'),
            ("fmap", "inputnode.fmap"),
            ("fmap_ref", "inputnode.fmap_ref"),
            ("fmap_coeff", "inputnode.fmap_coeff"),
            ("fmap_mask", "inputnode.fmap_mask"),
            ("fmap_id", "inputnode.fmap_id"),
            ("sdc_method", "inputnode.sdc_method"),
        ]),
    ])  # fmt:skip

    if config.workflow.level == "minimal":
        return workflow

    # Now that we're resampling and combining, multiecho matters
    multiecho = len(bold_series) > 2

    spaces = config.workflow.spaces
    nonstd_spaces = set(spaces.get_nonstandard())
    template_spaces = spaces.get_spaces(nonstandard=False, dim=(3,))
    freesurfer_spaces = spaces.get_fs_spaces()

    #
    # Resampling outputs workflow:
    #   - Resample to native
    #   - Save native outputs/echos only if requested
    #

    bold_native_wf = init_bold_native_wf(
        bold_series=bold_series,
        fieldmap_id=fieldmap_id,
        omp_nthreads=omp_nthreads,
    )
    bold_anat_wf = init_bold_volumetric_resample_wf(
        metadata=all_metadata[0],
        fieldmap_id=fieldmap_id if not multiecho else None,
        omp_nthreads=omp_nthreads,
        name='bold_anat_wf',
    )

    workflow.connect([
        (inputnode, bold_native_wf, [
            ("fmap_ref", "inputnode.fmap_ref"),
            ("fmap_coeff", "inputnode.fmap_coeff"),
            ("fmap_id", "inputnode.fmap_id"),
        ]),
        (inputnode, bold_anat_wf, [
            ("t1w_preproc", "inputnode.target_ref_file"),
            ("t1w_mask", "inputnode.target_mask"),
            ("fmap_ref", "inputnode.fmap_ref"),
            ("fmap_coeff", "inputnode.fmap_coeff"),
            ("fmap_id", "inputnode.fmap_id"),
        ]),
        (bold_fit_wf, bold_native_wf, [
            ("outputnode.coreg_boldref", "inputnode.boldref"),
            ("outputnode.bold_mask", "inputnode.bold_mask"),
            ("outputnode.motion_xfm", "inputnode.motion_xfm"),
            ("outputnode.boldref2fmap_xfm", "inputnode.boldref2fmap_xfm"),
            ("outputnode.dummy_scans", "inputnode.dummy_scans"),
        ]),
        (bold_fit_wf, bold_anat_wf, [
            ("outputnode.coreg_boldref", "inputnode.bold_ref_file"),
            ("outputnode.boldref2fmap_xfm", "inputnode.boldref2fmap_xfm"),
            ("outputnode.boldref2anat_xfm", "inputnode.boldref2anat_xfm"),
        ]),
        (bold_native_wf, bold_anat_wf, [
            ("outputnode.bold_minimal", "inputnode.bold_file"),
            ("outputnode.motion_xfm", "inputnode.motion_xfm"),
        ]),
    ])  # fmt:skip

    boldref_out = bool(nonstd_spaces.intersection(('func', 'run', 'bold', 'boldref', 'sbref')))
    boldref_out |= config.workflow.level == 'full'
    echos_out = multiecho and config.execution.me_output_echos

    if boldref_out or echos_out:
        ds_bold_native_wf = init_ds_bold_native_wf(
            bids_root=str(config.execution.bids_dir),
            output_dir=fmriprep_dir,
            bold_output=boldref_out,
            echo_output=echos_out,
            multiecho=multiecho,
            all_metadata=all_metadata,
        )
        ds_bold_native_wf.inputs.inputnode.source_files = bold_series

        workflow.connect([
            (bold_fit_wf, ds_bold_native_wf, [
                ('outputnode.bold_mask', 'inputnode.bold_mask'),
            ]),
            (bold_native_wf, ds_bold_native_wf, [
                ('outputnode.bold_native', 'inputnode.bold'),
                ('outputnode.bold_echos', 'inputnode.bold_echos'),
                ('outputnode.t2star_map', 'inputnode.t2star'),
            ]),
        ])  # fmt:skip

    if multiecho:
        t2s_reporting_wf = init_t2s_reporting_wf()

        ds_report_t2scomp = pe.Node(
            DerivativesDataSink(
                desc="t2scomp",
                datatype="figures",
                dismiss_entities=("echo",),
            ),
            name="ds_report_t2scomp",
            run_without_submitting=True,
        )

        ds_report_t2star_hist = pe.Node(
            DerivativesDataSink(
                desc="t2starhist",
                datatype="figures",
                dismiss_entities=("echo",),
            ),
            name="ds_report_t2star_hist",
            run_without_submitting=True,
        )

        workflow.connect([
            (inputnode, t2s_reporting_wf, [('t1w_dseg', 'inputnode.label_file')]),
            (bold_fit_wf, t2s_reporting_wf, [
                ('outputnode.boldref2anat_xfm', 'inputnode.boldref2anat_xfm'),
                ('outputnode.coreg_boldref', 'inputnode.boldref'),
            ]),
            (bold_native_wf, t2s_reporting_wf, [
                ('outputnode.t2star_map', 'inputnode.t2star_file'),
            ]),
            (t2s_reporting_wf, ds_report_t2scomp, [('outputnode.t2s_comp_report', 'in_file')]),
            (t2s_reporting_wf, ds_report_t2star_hist, [("outputnode.t2star_hist", "in_file")]),
        ])  # fmt:skip

    if config.workflow.run_reconall:
        vol2surf_fit_wf = init_vol2surf_fit_wf(
            filter_goodvoxels=config.workflow.project_goodvoxels,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb['filesize'],
        )

        ds_sampling_weights_wf = init_ds_registration_wf(
            bids_root=str(config.execution.bids_dir),
            output_dir=fmriprep_dir,
            source='T1w',
            dest='fsnative',
            name='ds_sampling_weights_wf',
        )
        ds_sampling_weights_wf.inputs.inputnode.source_files = bold_series

        workflow.connect([
            (inputnode, vol2surf_fit_wf, [
                ('midthickness', 'inputnode.midthickness'),
                ('white', 'inputnode.white'),
                ('pial', 'inputnode.pial'),
                ('anat_ribbon', 'inputnode.ribbon_mask'),
            ]),
            (bold_anat_wf, vol2surf_fit_wf, [('outputnode.bold_file', 'inputnode.bold_file')]),
            (vol2surf_fit_wf, ds_sampling_weights_wf, [
                ('outputnode.sampling_weights', 'inputnode.xform'),
            ]),
        ])  # fmt:skip

    if config.workflow.level == "resampling":
        return workflow

    # Full derivatives, including resampled BOLD series
    if nonstd_spaces.intersection(('anat', 'T1w')):
        ds_bold_t1_wf = init_ds_volumes_wf(
            bids_root=str(config.execution.bids_dir),
            output_dir=fmriprep_dir,
            multiecho=multiecho,
            metadata=all_metadata[0],
            name='ds_bold_t1_wf',
        )
        ds_bold_t1_wf.inputs.inputnode.source_files = bold_series
        ds_bold_t1_wf.inputs.inputnode.space = 'T1w'

        workflow.connect([
            (inputnode, ds_bold_t1_wf, [
                ('t1w_preproc', 'inputnode.ref_file'),
            ]),
            (bold_fit_wf, ds_bold_t1_wf, [
                ('outputnode.bold_mask', 'inputnode.bold_mask'),
                ('outputnode.coreg_boldref', 'inputnode.bold_ref'),
                ('outputnode.boldref2anat_xfm', 'inputnode.boldref2anat_xfm'),
            ]),
            (bold_native_wf, ds_bold_t1_wf, [('outputnode.t2star_map', 'inputnode.t2star')]),
            (bold_anat_wf, ds_bold_t1_wf, [('outputnode.bold_file', 'inputnode.bold')]),
        ])  # fmt:skip

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_report"):
            workflow.get_node(node).inputs.base_directory = fmriprep_dir
            workflow.get_node(node).inputs.source_file = bold_file

    return workflow


def init_func_preproc_wf(bold_file, has_fieldmap=False):
    """
    This workflow controls the functional preprocessing stages of *fMRIPrep*.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.tests import mock_config
            from fmriprep import config
            from fmriprep.workflows.bold.base import init_func_preproc_wf
            with mock_config():
                bold_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
                wf = init_func_preproc_wf(str(bold_file))

    Parameters
    ----------
    bold_file
        Path to NIfTI file (single echo) or list of paths to NIfTI files (multi-echo)
    has_fieldmap : :obj:`bool`
        Signals the workflow to use inputnode fieldmap files

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    t1w_preproc
        Bias-corrected structural template image
    t1w_mask
        Mask of the skull-stripped template image
    t1w_dseg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    t1w_aseg
        Segmentation of structural image, done with FreeSurfer.
    t1w_aparc
        Parcellation of structural image, done with FreeSurfer.
    t1w_tpms
        List of tissue probability maps in T1w space
    template
        List of templates to target
    anat2std_xfm
        List of transform files, collated with templates
    std2anat_xfm
        List of inverse transform files, collated with templates
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    fsnative2t1w_xfm
        LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w

    Outputs
    -------
    bold_t1
        BOLD series, resampled to T1w space
    bold_t1_ref
        BOLD reference image, resampled to T1w space
    bold2anat_xfm
        Affine transform from BOLD reference space to T1w space
    anat2bold_xfm
        Affine transform from T1w space to BOLD reference space
    hmc_xforms
        Affine transforms for each BOLD volume to the BOLD reference
    bold_mask_t1
        BOLD series mask in T1w space
    bold_aseg_t1
        FreeSurfer ``aseg`` resampled to match ``bold_t1``
    bold_aparc_t1
        FreeSurfer ``aparc+aseg`` resampled to match ``bold_t1``
    bold_std
        BOLD series, resampled to template space
    bold_std_ref
        BOLD reference image, resampled to template space
    bold_mask_std
        BOLD series mask in template space
    bold_aseg_std
        FreeSurfer ``aseg`` resampled to match ``bold_std``
    bold_aparc_std
        FreeSurfer ``aparc+aseg`` resampled to match ``bold_std``
    bold_native
        BOLD series, with distortion corrections applied (native space)
    bold_native_ref
        BOLD reference image in native space
    bold_mask_native
        BOLD series mask in native space
    bold_echos_native
        Per-echo BOLD series, with distortion corrections applied
    bold_cifti
        BOLD CIFTI image
    cifti_metadata
        Path of metadata files corresponding to ``bold_cifti``.
    surfaces
        BOLD series, resampled to FreeSurfer surfaces
    t2star_bold
        Estimated T2\\* map in BOLD native space
    t2star_t1
        Estimated T2\\* map in T1w space
    t2star_std
        Estimated T2\\* map in template space
    confounds
        TSV of confounds
    confounds_metadata
        Confounds metadata dictionary

    See Also
    --------

    * :py:func:`~niworkflows.func.util.init_bold_reference_wf`
    * :py:func:`~fmriprep.workflows.bold.stc.init_bold_stc_wf`
    * :py:func:`~fmriprep.workflows.bold.hmc.init_bold_hmc_wf`
    * :py:func:`~fmriprep.workflows.bold.t2s.init_bold_t2s_wf`
    * :py:func:`~fmriprep.workflows.bold.t2s.init_t2s_reporting_wf`
    * :py:func:`~fmriprep.workflows.bold.registration.init_bold_t1_trans_wf`
    * :py:func:`~fmriprep.workflows.bold.registration.init_bold_reg_wf`
    * :py:func:`~fmriprep.workflows.bold.confounds.init_bold_confs_wf`
    * :py:func:`~fmriprep.workflows.bold.resampling.init_bold_std_trans_wf`
    * :py:func:`~fmriprep.workflows.bold.resampling.init_bold_preproc_trans_wf`
    * :py:func:`~fmriprep.workflows.bold.resampling.init_bold_surf_wf`
    * :py:func:`~sdcflows.workflows.fmap.init_fmap_wf`
    * :py:func:`~sdcflows.workflows.pepolar.init_pepolar_unwarp_wf`
    * :py:func:`~sdcflows.workflows.phdiff.init_phdiff_wf`
    * :py:func:`~sdcflows.workflows.syn.init_syn_sdc_wf`
    * :py:func:`~sdcflows.workflows.unwarp.init_sdc_unwarp_wf`

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.func.util import init_bold_reference_wf
    from niworkflows.interfaces.nibabel import ApplyMask
    from niworkflows.interfaces.reportlets.registration import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )
    from niworkflows.interfaces.utility import KeySelect

    # Have some options handy
    omp_nthreads = config.nipype.omp_nthreads
    freesurfer = config.workflow.run_reconall
    spaces = config.workflow.spaces
    fmriprep_dir = str(config.execution.fmriprep_dir)
    freesurfer_spaces = spaces.get_fs_spaces()

    ref_file = bold_file
    wf_name = _get_wf_name(ref_file, "func_preproc")

    # Build workflow
    workflow = Workflow(name=wf_name)
    workflow.__postdesc__ = """\
All resamplings can be performed with *a single interpolation
step* by composing all the pertinent transformations (i.e. head-motion
transform matrices, susceptibility distortion correction when available,
and co-registrations to anatomical and output spaces).
Gridded (volumetric) resamplings were performed using `antsApplyTransforms` (ANTs),
configured with Lanczos interpolation to minimize the smoothing
effects of other kernels [@lanczos].
Non-gridded (surface) resamplings were performed using `mri_vol2surf`
(FreeSurfer).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "subjects_dir",
                "subject_id",
                "t1w_preproc",
                "t1w_mask",
                "t1w_dseg",
                "t1w_tpms",
                "t1w_aseg",
                "t1w_aparc",
                "anat2std_xfm",
                "std2anat_xfm",
                "template",
                "anat_ribbon",
                "fsnative2t1w_xfm",
                "surfaces",
                "morphometrics",
                "sphere_reg_fsLR",
                "fmap",
                "fmap_ref",
                "fmap_coeff",
                "fmap_mask",
                "fmap_id",
                "sdc_method",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.bold_file = bold_file

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_t1",
                "bold_t1_ref",
                "bold2anat_xfm",
                "anat2bold_xfm",
                "hmc_xforms",
                "bold_mask_t1",
                "bold_aseg_t1",
                "bold_aparc_t1",
                "bold_std",
                "bold_std_ref",
                "bold_mask_std",
                "bold_aseg_std",
                "bold_aparc_std",
                "bold_native",
                "bold_native_ref",
                "bold_mask_native",
                "bold_echos_native",
                "bold_cifti",
                "cifti_metadata",
                "surfaces",
                "t2star_bold",
                "t2star_t1",
                "t2star_std",
                "confounds",
                "confounds_metadata",
                "weights_text",
            ]
        ),
        name="outputnode",
    )

    # get confounds
    bold_confounds_wf = init_bold_confs_wf(
        mem_gb=mem_gb["largemem"],
        metadata=metadata,
        freesurfer=freesurfer,
        regressors_all_comps=config.workflow.regressors_all_comps,
        regressors_fd_th=config.workflow.regressors_fd_th,
        regressors_dvars_th=config.workflow.regressors_dvars_th,
        name="bold_confounds_wf",
    )
    bold_confounds_wf.get_node("inputnode").inputs.t1_transform_flags = [False]

    if spaces.get_spaces(nonstandard=False, dim=(3,)):
        # Apply transforms in 1 shot
        bold_std_trans_wf = init_bold_std_trans_wf(
            freesurfer=freesurfer,
            mem_gb=mem_gb["resampled"],
            omp_nthreads=omp_nthreads,
            spaces=spaces,
            multiecho=multiecho,
            name="bold_std_trans_wf",
            use_compression=not config.execution.low_mem,
        )
        bold_std_trans_wf.inputs.inputnode.fieldwarp = "identity"

        # fmt:off
        workflow.connect([
            (inputnode, bold_std_trans_wf, [
                ("template", "inputnode.templates"),
                ("anat2std_xfm", "inputnode.anat2std_xfm"),
                ("bold_file", "inputnode.name_source"),
                ("t1w_aseg", "inputnode.bold_aseg"),
                ("t1w_aparc", "inputnode.bold_aparc"),
            ]),
            (bold_final, bold_std_trans_wf, [
                ("mask", "inputnode.bold_mask"),
                ("t2star", "inputnode.t2star"),
            ]),
            (bold_reg_wf, bold_std_trans_wf, [
                ("outputnode.itk_bold_to_t1", "inputnode.itk_bold_to_t1"),
            ]),
            (bold_std_trans_wf, outputnode, [
                ("outputnode.bold_std", "bold_std"),
                ("outputnode.bold_std_ref", "bold_std_ref"),
                ("outputnode.bold_mask_std", "bold_mask_std"),
            ]),
        ])
        # fmt:on

        if freesurfer:
            # fmt:off
            workflow.connect([
                (bold_std_trans_wf, func_derivatives_wf, [
                    ("outputnode.bold_aseg_std", "inputnode.bold_aseg_std"),
                    ("outputnode.bold_aparc_std", "inputnode.bold_aparc_std"),
                ]),
                (bold_std_trans_wf, outputnode, [
                    ("outputnode.bold_aseg_std", "bold_aseg_std"),
                    ("outputnode.bold_aparc_std", "bold_aparc_std"),
                ]),
            ])
            # fmt:on

        if not multiecho:
            # fmt:off
            workflow.connect([
                (bold_split, bold_std_trans_wf, [("out_files", "inputnode.bold_split")]),
                (bold_hmc_wf, bold_std_trans_wf, [
                    ("outputnode.xforms", "inputnode.hmc_xforms"),
                ]),
            ])
            # fmt:on
        else:
            # fmt:off
            workflow.connect([
                (split_opt_comb, bold_std_trans_wf, [("out_files", "inputnode.bold_split")]),
                (bold_std_trans_wf, outputnode, [("outputnode.t2star_std", "t2star_std")]),
            ])
            # fmt:on

            # Already applied in bold_bold_trans_wf, which inputs to bold_t2s_wf
            bold_std_trans_wf.inputs.inputnode.hmc_xforms = "identity"

        # fmt:off
        # func_derivatives_wf internally parametrizes over snapshotted spaces.
        workflow.connect([
            (bold_std_trans_wf, func_derivatives_wf, [
                ("outputnode.template", "inputnode.template"),
                ("outputnode.spatial_reference", "inputnode.spatial_reference"),
                ("outputnode.bold_std_ref", "inputnode.bold_std_ref"),
                ("outputnode.bold_std", "inputnode.bold_std"),
                ("outputnode.bold_mask_std", "inputnode.bold_mask_std"),
            ]),
        ])
        # fmt:on

    # SURFACES ##################################################################################
    # Freesurfer
    if freesurfer and freesurfer_spaces:
        config.loggers.workflow.debug("Creating BOLD surface-sampling workflow.")
        bold_surf_wf = init_bold_surf_wf(
            mem_gb=mem_gb["resampled"],
            surface_spaces=freesurfer_spaces,
            medial_surface_nan=config.workflow.medial_surface_nan,
            name="bold_surf_wf",
        )
        # fmt:off
        workflow.connect([
            (inputnode, bold_surf_wf, [
                ("subjects_dir", "inputnode.subjects_dir"),
                ("subject_id", "inputnode.subject_id"),
                ("fsnative2t1w_xfm", "inputnode.fsnative2t1w_xfm"),
            ]),
            (bold_t1_trans_wf, bold_surf_wf, [("outputnode.bold_t1", "inputnode.source_file")]),
            (bold_surf_wf, outputnode, [("outputnode.surfaces", "surfaces")]),
            (bold_surf_wf, func_derivatives_wf, [("outputnode.target", "inputnode.surf_refs")]),
        ])
        # fmt:on

    # CIFTI output
    if config.workflow.cifti_output:
        from .resampling import init_bold_fsLR_resampling_wf, init_bold_grayords_wf

        bold_fsLR_resampling_wf = init_bold_fsLR_resampling_wf(
            estimate_goodvoxels=project_goodvoxels,
            grayord_density=config.workflow.cifti_output,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb["resampled"],
        )

        bold_grayords_wf = init_bold_grayords_wf(
            grayord_density=config.workflow.cifti_output,
            mem_gb=mem_gb["resampled"],
            repetition_time=metadata["RepetitionTime"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, bold_fsLR_resampling_wf, [
                ("surfaces", "inputnode.surfaces"),
                ("morphometrics", "inputnode.morphometrics"),
                ("sphere_reg_fsLR", "inputnode.sphere_reg_fsLR"),
                ("anat_ribbon", "inputnode.anat_ribbon"),
            ]),
            (bold_t1_trans_wf, bold_fsLR_resampling_wf, [
                ("outputnode.bold_t1", "inputnode.bold_file"),
            ]),
            (bold_std_trans_wf, bold_grayords_wf, [
                ("outputnode.bold_std", "inputnode.bold_std"),
                ("outputnode.spatial_reference", "inputnode.spatial_reference"),
            ]),
            (bold_fsLR_resampling_wf, bold_grayords_wf, [
                ("outputnode.bold_fsLR", "inputnode.bold_fsLR"),
            ]),
            (bold_fsLR_resampling_wf, func_derivatives_wf, [
                ("outputnode.goodvoxels_mask", "inputnode.goodvoxels_mask"),
            ]),
            (bold_fsLR_resampling_wf, outputnode, [
                ("outputnode.weights_text", "weights_text"),
            ]),
            (bold_grayords_wf, outputnode, [
                ("outputnode.cifti_bold", "bold_cifti"),
                ("outputnode.cifti_metadata", "cifti_metadata"),
            ]),
        ])
        # fmt:on

    if spaces.get_spaces(nonstandard=False, dim=(3,)):
        carpetplot_wf = init_carpetplot_wf(
            mem_gb=mem_gb["resampled"],
            metadata=metadata,
            cifti_output=config.workflow.cifti_output,
            name="carpetplot_wf",
        )

        # Xform to "MNI152NLin2009cAsym" is always computed.
        carpetplot_select_std = pe.Node(
            KeySelect(fields=["std2anat_xfm"], key="MNI152NLin2009cAsym"),
            name="carpetplot_select_std",
            run_without_submitting=True,
        )

        if config.workflow.cifti_output:
            # fmt:off
            workflow.connect(
                bold_grayords_wf, "outputnode.cifti_bold", carpetplot_wf, "inputnode.cifti_bold",
            )
            # fmt:on

        def _last(inlist):
            return inlist[-1]

        # fmt:off
        workflow.connect([
            (initial_boldref_wf, carpetplot_wf, [
                ("outputnode.skip_vols", "inputnode.dummy_scans"),
            ]),
            (inputnode, carpetplot_select_std, [("std2anat_xfm", "std2anat_xfm"),
                                                ("template", "keys")]),
            (carpetplot_select_std, carpetplot_wf, [
                ("std2anat_xfm", "inputnode.std2anat_xfm"),
            ]),
            (bold_final, carpetplot_wf, [
                ("bold", "inputnode.bold"),
                ("mask", "inputnode.bold_mask"),
            ]),
            (bold_reg_wf, carpetplot_wf, [
                ("outputnode.itk_t1_to_bold", "inputnode.t1_bold_xform"),
            ]),
            (bold_confounds_wf, carpetplot_wf, [
                ("outputnode.confounds_file", "inputnode.confounds_file"),
                ("outputnode.crown_mask", "inputnode.crown_mask"),
                (("outputnode.acompcor_masks", _last), "inputnode.acompcor_mask"),
            ]),
        ])
        # fmt:on

    # REPORTING ############################################################
    ds_report_summary = pe.Node(
        DerivativesDataSink(desc="summary", datatype="figures", dismiss_entities=("echo",)),
        name="ds_report_summary",
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    ds_report_validation = pe.Node(
        DerivativesDataSink(desc="validation", datatype="figures", dismiss_entities=("echo",)),
        name="ds_report_validation",
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    # fmt:off
    workflow.connect([
        (summary, ds_report_summary, [("out_report", "in_file")]),
        (initial_boldref_wf, ds_report_validation, [("outputnode.validation_report", "in_file")]),
    ])
    # fmt:on

    return workflow


def _create_mem_gb(bold_fname):
    img = nb.load(bold_fname)
    nvox = int(np.prod(img.shape, dtype='u8'))
    # Assume tools will coerce to 8-byte floats to be safe
    bold_size_gb = 8 * nvox / (1024**3)
    bold_tlen = 1 if img.ndim < 4 else img.shape[3]
    mem_gb = {
        "filesize": bold_size_gb,
        "resampled": bold_size_gb * 4,
        "largemem": bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return bold_tlen, mem_gb


def _get_wf_name(bold_fname, prefix):
    """
    Derive the workflow name for supplied BOLD file.

    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_bold.nii.gz", "bold")
    'bold_task_nback_wf'
    >>> _get_wf_name(
    ...     "/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz",
    ...     "preproc",
    ... )
    'preproc_task_nback_run_01_echo_1_wf'

    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = "_".join(fname.split("_")[1:-1])
    return f'{prefix}_{fname_nosub.replace("-", "_")}_wf'


def _to_join(in_file, join_file):
    """Join two tsv files if the join_file is not ``None``."""
    from niworkflows.interfaces.utility import JoinTSVColumns

    if join_file is None:
        return in_file
    res = JoinTSVColumns(in_file=in_file, join_file=join_file).run()
    return res.outputs.out_file


def extract_entities(file_list):
    """
    Return a dictionary of common entities given a list of files.

    Examples
    --------
    >>> extract_entities("sub-01/anat/sub-01_T1w.nii.gz")
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_T1w.nii.gz"] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_run-1_T1w.nii.gz",
    ...                   "sub-01/anat/sub-01_run-2_T1w.nii.gz"])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}

    """
    from collections import defaultdict

    from bids.layout import parse_file_entities

    entities = defaultdict(list)
    for e, v in [
        ev_pair for f in listify(file_list) for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist

    return {k: _unique(v) for k, v in entities.items()}


def get_img_orientation(imgf):
    """Return the image orientation as a string"""
    img = nb.load(imgf)
    return "".join(nb.aff2axcodes(img.affine))
