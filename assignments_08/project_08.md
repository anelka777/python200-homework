# Week 8 Project — Supabase Setup & Cloud Cost Analysis

## Part A: Supabase Setup

The `python200` Supabase project was created successfully. Both the `weather_raw` and `weather_enriched` tables are visible in the Table Editor with the expected columns, and Row Level Security has been disabled on both as instructed.

## Part B: Cloud Cost Analysis

**Scenario A — Lightweight compute (t3.micro, on-demand, 160 hours/month, US East N. Virginia):**
$1.66/month ($19.92/year). This is basically negligible — a small, part-time general-purpose instance costs almost nothing on-demand.

**Scenario B — Heavy analytics workload (US East N. Virginia):**
- EC2 p3.2xlarge (GPU instance, on-demand, 730 hours/month — full-time): $2,233.80/month
- RDS db.m5.large (PostgreSQL, Single-AZ, on-demand, 730 hours/month): $181.59/month
- S3 Standard storage (1 TB): $23.55/month
- **Total: ~$2,438.94/month (~$29,267/year)**

The gap between the two scenarios was surprising — Scenario B is over 1,400x more expensive than Scenario A. The GPU instance alone accounts for the vast majority of that cost (~92% of the total), running 24/7 for a full month on a single p3.2xlarge instance.

**Comparison:** The huge cost difference between the two scenarios shows that a GPU instance is only worth it when it's actually being used for GPU-bound work — running one 24/7 "just in case" is extremely expensive per month. It makes far more sense to spin a GPU instance up only for the hours it's actively needed (e.g., during a training job) and shut it down immediately after, rather than leaving it running continuously like a general-purpose web server.

## Video

[Video link](https://youtu.be/2b4IAj65uCs)

The video shows:
- The Supabase dashboard — project overview, both tables in the Table Editor, and the API settings page with the Project URL and anon key.
- The AWS Pricing Calculator — completed estimates for both scenarios, with a short walkthrough of what each scenario costs and what was surprising.