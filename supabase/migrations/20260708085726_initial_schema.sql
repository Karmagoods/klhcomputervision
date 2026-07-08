-- Initial schema for KLH Computer Vision.
-- Compatible with the Supabase CLI migration runner.

create extension if not exists pgcrypto with schema extensions;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  full_name text,
  avatar_url text,
  company_name text,
  plan text not null default 'free',
  settings jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  constraint profiles_plan_check check (plan in ('free', 'starter', 'pro', 'enterprise')),
  constraint profiles_email_format_check check (email is null or position('@' in email) > 1),
  constraint profiles_settings_object_check check (jsonb_typeof(settings) = 'object')
);

create table public.computer_vision_projects (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  name text not null,
  description text,
  project_type text not null default 'general',
  status text not null default 'active',
  model_provider text,
  model_name text,
  configuration jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  constraint computer_vision_projects_name_not_blank check (length(btrim(name)) > 0),
  constraint computer_vision_projects_status_check check (status in ('active', 'archived', 'deleted')),
  constraint computer_vision_projects_type_check check (
    project_type in ('general', 'object_detection', 'face_detection', 'motion_detection', 'classification', 'segmentation')
  ),
  constraint computer_vision_projects_configuration_object_check check (jsonb_typeof(configuration) = 'object'),
  constraint computer_vision_projects_id_owner_unique unique (id, owner_id)
);

create table public.analyses (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null,
  owner_id uuid not null references public.profiles(id) on delete cascade,
  status text not null default 'queued',
  analysis_type text not null,
  input_source text,
  model_provider text,
  model_name text,
  parameters jsonb not null default '{}'::jsonb,
  results jsonb not null default '{}'::jsonb,
  confidence_score numeric(5,4),
  error_message text,
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  constraint analyses_status_check check (status in ('queued', 'processing', 'completed', 'failed', 'cancelled')),
  constraint analyses_type_not_blank check (length(btrim(analysis_type)) > 0),
  constraint analyses_confidence_score_check check (confidence_score is null or confidence_score between 0 and 1),
  constraint analyses_parameters_object_check check (jsonb_typeof(parameters) = 'object'),
  constraint analyses_results_object_check check (jsonb_typeof(results) = 'object'),
  constraint analyses_id_owner_unique unique (id, owner_id),
  constraint analyses_project_owner_fk foreign key (project_id, owner_id)
    references public.computer_vision_projects(id, owner_id) on delete cascade
);

create table public.image_metadata (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  project_id uuid references public.computer_vision_projects(id) on delete cascade,
  analysis_id uuid references public.analyses(id) on delete set null,
  storage_bucket text not null default 'images',
  storage_path text not null,
  original_filename text,
  content_type text,
  file_size_bytes bigint,
  width integer,
  height integer,
  checksum_sha256 text,
  capture_timestamp timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  constraint image_metadata_storage_path_not_blank check (length(btrim(storage_path)) > 0),
  constraint image_metadata_file_size_check check (file_size_bytes is null or file_size_bytes >= 0),
  constraint image_metadata_width_check check (width is null or width > 0),
  constraint image_metadata_height_check check (height is null or height > 0),
  constraint image_metadata_checksum_sha256_check check (checksum_sha256 is null or checksum_sha256 ~ '^[a-f0-9]{64}$'),
  constraint image_metadata_metadata_object_check check (jsonb_typeof(metadata) = 'object'),
  constraint image_metadata_owner_storage_path_unique unique (owner_id, storage_bucket, storage_path)
);

create table public.analytics_events (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  project_id uuid references public.computer_vision_projects(id) on delete set null,
  analysis_id uuid references public.analyses(id) on delete set null,
  event_name text not null,
  event_category text,
  session_id text,
  page_path text,
  properties jsonb not null default '{}'::jsonb,
  user_agent text,
  ip_address inet,
  occurred_at timestamptz not null default now(),
  created_at timestamptz not null default now(),

  constraint analytics_events_event_name_not_blank check (length(btrim(event_name)) > 0),
  constraint analytics_events_properties_object_check check (jsonb_typeof(properties) = 'object')
);

create index computer_vision_projects_owner_id_idx on public.computer_vision_projects(owner_id);
create index computer_vision_projects_owner_status_idx on public.computer_vision_projects(owner_id, status);
create index computer_vision_projects_created_at_idx on public.computer_vision_projects(created_at desc);

create index analyses_project_id_idx on public.analyses(project_id);
create index analyses_owner_id_idx on public.analyses(owner_id);
create index analyses_owner_status_idx on public.analyses(owner_id, status);
create index analyses_created_at_idx on public.analyses(created_at desc);

create index image_metadata_owner_id_idx on public.image_metadata(owner_id);
create index image_metadata_project_id_idx on public.image_metadata(project_id);
create index image_metadata_analysis_id_idx on public.image_metadata(analysis_id);
create index image_metadata_created_at_idx on public.image_metadata(created_at desc);

create index analytics_events_owner_id_idx on public.analytics_events(owner_id);
create index analytics_events_project_id_idx on public.analytics_events(project_id);
create index analytics_events_analysis_id_idx on public.analytics_events(analysis_id);
create index analytics_events_name_occurred_at_idx on public.analytics_events(event_name, occurred_at desc);
create index analytics_events_owner_occurred_at_idx on public.analytics_events(owner_id, occurred_at desc);

create trigger set_profiles_updated_at
before update on public.profiles
for each row execute function public.set_updated_at();

create trigger set_computer_vision_projects_updated_at
before update on public.computer_vision_projects
for each row execute function public.set_updated_at();

create trigger set_analyses_updated_at
before update on public.analyses
for each row execute function public.set_updated_at();

create trigger set_image_metadata_updated_at
before update on public.image_metadata
for each row execute function public.set_updated_at();

alter table public.profiles enable row level security;
alter table public.computer_vision_projects enable row level security;
alter table public.analyses enable row level security;
alter table public.image_metadata enable row level security;
alter table public.analytics_events enable row level security;

create policy "Users can view their own profile"
on public.profiles for select
to authenticated
using (id = auth.uid());

create policy "Users can create their own profile"
on public.profiles for insert
to authenticated
with check (id = auth.uid());

create policy "Users can update their own profile"
on public.profiles for update
to authenticated
using (id = auth.uid())
with check (id = auth.uid());

create policy "Users can delete their own profile"
on public.profiles for delete
to authenticated
using (id = auth.uid());

create policy "Users can view their own projects"
on public.computer_vision_projects for select
to authenticated
using (owner_id = auth.uid());

create policy "Users can create their own projects"
on public.computer_vision_projects for insert
to authenticated
with check (owner_id = auth.uid());

create policy "Users can update their own projects"
on public.computer_vision_projects for update
to authenticated
using (owner_id = auth.uid())
with check (owner_id = auth.uid());

create policy "Users can delete their own projects"
on public.computer_vision_projects for delete
to authenticated
using (owner_id = auth.uid());

create policy "Users can view their own analyses"
on public.analyses for select
to authenticated
using (owner_id = auth.uid());

create policy "Users can create their own analyses"
on public.analyses for insert
to authenticated
with check (
  owner_id = auth.uid()
  and exists (
    select 1
    from public.computer_vision_projects
    where computer_vision_projects.id = analyses.project_id
      and computer_vision_projects.owner_id = auth.uid()
  )
);

create policy "Users can update their own analyses"
on public.analyses for update
to authenticated
using (owner_id = auth.uid())
with check (
  owner_id = auth.uid()
  and exists (
    select 1
    from public.computer_vision_projects
    where computer_vision_projects.id = analyses.project_id
      and computer_vision_projects.owner_id = auth.uid()
  )
);

create policy "Users can delete their own analyses"
on public.analyses for delete
to authenticated
using (owner_id = auth.uid());

create policy "Users can view their own image metadata"
on public.image_metadata for select
to authenticated
using (owner_id = auth.uid());

create policy "Users can create their own image metadata"
on public.image_metadata for insert
to authenticated
with check (
  owner_id = auth.uid()
  and (
    project_id is null
    or exists (
      select 1
      from public.computer_vision_projects
      where computer_vision_projects.id = image_metadata.project_id
        and computer_vision_projects.owner_id = auth.uid()
    )
  )
  and (
    analysis_id is null
    or exists (
      select 1
      from public.analyses
      where analyses.id = image_metadata.analysis_id
        and analyses.owner_id = auth.uid()
    )
  )
);

create policy "Users can update their own image metadata"
on public.image_metadata for update
to authenticated
using (owner_id = auth.uid())
with check (
  owner_id = auth.uid()
  and (
    project_id is null
    or exists (
      select 1
      from public.computer_vision_projects
      where computer_vision_projects.id = image_metadata.project_id
        and computer_vision_projects.owner_id = auth.uid()
    )
  )
  and (
    analysis_id is null
    or exists (
      select 1
      from public.analyses
      where analyses.id = image_metadata.analysis_id
        and analyses.owner_id = auth.uid()
    )
  )
);

create policy "Users can delete their own image metadata"
on public.image_metadata for delete
to authenticated
using (owner_id = auth.uid());

create policy "Users can view their own analytics events"
on public.analytics_events for select
to authenticated
using (owner_id = auth.uid());

create policy "Users can create their own analytics events"
on public.analytics_events for insert
to authenticated
with check (
  owner_id = auth.uid()
  and (
    project_id is null
    or exists (
      select 1
      from public.computer_vision_projects
      where computer_vision_projects.id = analytics_events.project_id
        and computer_vision_projects.owner_id = auth.uid()
    )
  )
  and (
    analysis_id is null
    or exists (
      select 1
      from public.analyses
      where analyses.id = analytics_events.analysis_id
        and analyses.owner_id = auth.uid()
    )
  )
);

grant usage on schema public to authenticated;
grant select, insert, update, delete on public.profiles to authenticated;
grant select, insert, update, delete on public.computer_vision_projects to authenticated;
grant select, insert, update, delete on public.analyses to authenticated;
grant select, insert, update, delete on public.image_metadata to authenticated;
grant select, insert on public.analytics_events to authenticated;
