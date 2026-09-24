# Data Directory

This directory is reserved for derived or cached data files used by the application.

- Keep the original `resume dataset.zip` in the project root unchanged.
- The job dataset source is Hugging Face: `batuhanmtl/job-skill-set`.
- Do not add invented job records or columns.
- Any downloaded or transformed copy must retain the source fields:
  `job_id`, `category`, `job_title`, `job_description`, and `job_skill_set`.