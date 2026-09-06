# Week 8 Warmup — Cloud Concepts & Cloud Landscape

## Part 1: Cloud Concepts

### Question 1 — Core economic model of cloud computing

Cloud computing is a pay-as-you-go rental model: instead of buying and maintaining your own servers (a large upfront capital expense), you rent compute, storage, and networking from a provider and pay only for what you actually use. Owning your own servers means paying upfront for hardware that has to cover your *peak* possible load, even if it sits idle most of the time, plus ongoing costs for maintenance, cooling, and eventual replacement. With cloud, you can scale resources up or down on demand and never pay for idle capacity.

### Question 2 — Vertical vs. horizontal scaling

- **Vertical scaling** means making a single machine bigger — more CPU, more RAM, a faster GPU. It's simpler to implement, but it has a hard ceiling (there's a limit to how big one machine can get).
- **Horizontal scaling** means adding more machines and splitting the work across them. It's more complex to set up (you need to coordinate/balance work across machines), but it scales much further and handles unpredictable spikes in demand well.

**Example — vertical scaling:** a data scientist whose model training is slow because the machine doesn't have enough GPU/RAM would benefit from a single, more powerful machine — there's one job to speed up, not more instances to add.

**Example — horizontal scaling:** a web app suddenly getting way more traffic than usual needs more servers running in parallel to handle the concurrent requests, not one bigger server.

**Scenarios:**

1. A web app going from 1,000 to 100,000 users/day needs **horizontal scaling** — the workload is many simultaneous requests, which is best handled by spreading load across multiple servers.
2. A data scientist wanting a faster GPU and more RAM for a single training job needs **vertical scaling** — it's one job on one machine that just needs to be more powerful.
3. A data pipeline going from 10 to 10,000 files per run, where the work can be split across machines, needs **horizontal scaling** — it's a parallelizable workload well suited to distributing across many machines.

### Question 3 — IaaS, PaaS, SaaS, BaaS

**Classification:**

- **Gmail — SaaS.** It's a finished application I just log into and use; I never think about servers, infrastructure, or code.
- **Azure Virtual Machines — IaaS.** I get raw compute — I choose the OS, install software, and manage configuration and updates myself.
- **AWS S3 — IaaS.** It's a raw infrastructure primitive (object storage) with no application layer built on top.
- **GitHub Codespaces — PaaS.** It gives me a ready-to-use development environment; I don't manage the underlying servers, just my code and configuration.
- **Snowflake — PaaS.** It's a managed platform where I write queries and manage data, but the underlying infrastructure (servers, scaling) is handled for me.
- **Supabase — BaaS.** I get a pre-wired backend (database, auth, storage, API) that I just connect to, without provisioning or managing infrastructure myself.

**In my own words:**

- **IaaS (Infrastructure as a Service)** gives you raw computing resources — a virtual machine, storage, networking — and you're responsible for everything from the operating system up: installing software, configuring the environment, applying security updates. Example: Azure Virtual Machines. As the developer, I manage the OS, runtime, and everything I install on top of it.
- **PaaS (Platform as a Service)** manages the infrastructure for you — you just bring your code. The platform handles running it, scaling it, and keeping the machine healthy. Example: GitHub Codespaces / Snowflake. As the developer, I'm responsible for my code/queries and configuration, not the servers underneath.
- **SaaS (Software as a Service)** is a finished application that someone else builds, hosts, and maintains — you just log in and use it. Example: Gmail. As the developer/user, I'm not responsible for managing anything technical at all.

### Question 4 — Managed data platforms vs. cloud providers directly

A managed data platform like Databricks or Snowflake is a layer built *on top of* a cloud provider (AWS, GCP, or Azure) that pre-wires infrastructure specifically for data and analytics workloads, instead of making you assemble compute, storage, and orchestration yourself. Databricks, for example, actually runs on AWS/GCP/Azure under the hood — it's not a separate cloud.

**What you gain:** much faster setup for large-scale data processing or ML — you're not manually wiring together compute, storage, and pipelines.
**What you give up:** some flexibility and control (you're working within the platform's abstractions), and potentially higher cost compared to configuring the raw cloud services yourself.

### Question 5 — When cloud is probably not the right choice

If your dataset fits comfortably on a single machine and you don't have heavy compute demands, local processing is often faster and cheaper — this is especially true for an initial prototype, where the overhead of cloud setup isn't worth it yet.

---

## Part 2: Cloud Landscape

### Question 1 — The three hyperscalers

- **AWS (Amazon Web Services)** — the oldest and largest provider, with the broadest service catalog of any cloud. It's the most likely candidate for large enterprises, startups, and nonprofits with engineering staff.
- **GCP (Google Cloud Platform)** — strongest in data and machine learning, building on Google's history with distributed systems (e.g., BigQuery). It's often preferred for large-scale analytics or ML infrastructure work.
- **Microsoft Azure** — dominant in enterprise and government settings because of its deep integration with Windows, Active Directory, and Microsoft 365. Organizations already on Microsoft agreements tend to default to Azure.

### Question 2 — Why this course switched from Azure to Supabase

1. **Access** — Azure requires organizational provisioning (joining a tenant, waiting for an invite, configuring auth), which can block students for days if something goes wrong. Supabase accounts are self-provisioned in under two minutes, and the free tier covers everything needed for the course.
2. **Pedagogical fit** — Azure Blob Storage (the Azure service the course used) stores data as opaque files organized by path. Supabase is a real relational database — rows, columns, SQL — which is a much more transferable skill for data work in general.
3. **Pipeline coherence** — the ETL pipeline built in weeks 9–11 has a raw zone and an enriched zone, which map naturally onto two related tables in Supabase. That makes each stage of the pipeline easy to inspect and debug by just querying the table.

**My reflection:** This suggests that when evaluating a cloud tool for a new project, "more powerful" or "more well-known" isn't the right first question — the better questions are how much friction there is to actually get started (onboarding, access, setup time) and how naturally the tool's data model matches the shape of the problem you're solving. A tool that's a great conceptual fit and low-friction to set up can be more valuable early on than the most powerful/general option.

### Question 3 — Service category + provider for each scenario

1. **Storing 10 TB of images, retrieved by filename** → **Object storage** → AWS S3
2. **Running an ML training job on a GPU for four hours, then shutting it down** → **Compute (GPU instance)** → AWS EC2 (e.g., a p3 instance)
3. **Hosting a web API that auto-scales up and down with traffic** → **Serverless compute** → AWS Lambda (Supabase Edge Functions is also a valid answer here)
4. **Sending structured data to an LLM and getting a text response back** → **LLM API** → AWS Bedrock (Azure OpenAI or GCP Vertex AI would also fit)

### Question 4 — A multi-provider stack

**My project:** A pipeline that pulls daily weather data from a public API, stores it, runs a simple model/LLM step to classify whether conditions are good for running, and stores the enriched result.

**Stack:** an external **Weather API** for the raw data source, **Supabase** as the relational database for the raw and enriched tables, and the **OpenAI API** directly for the LLM enrichment step — three different providers/products from the taxonomy, not one.

**Is there a benefit to consolidating to one provider?** Yes — a single provider means one bill, one set of credentials/access controls, and less operational complexity to manage. **What would I give up?** The ability to pick the best tool for each specific job — for example, Supabase's relational-database ergonomics versus a general-purpose cloud provider's more generic (and more complex) database offering — and I'd risk more vendor lock-in.