import json
import os
from typing import Any, Dict, Optional, List, Tuple

from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from .logging_utils import get_logger
from .models import ParsedURL, URLCategory

logger = get_logger()


class HuggingFaceAPI:
    # wrapper for Hugging Face Hub API operations (synchronous)

    def __init__(self):
        self.api = HfApi()
        self.token = os.getenv("HF_TOKEN")
        self.timeout = 30.0

    def get_model_file_urls(
        self,
        model_id: str,
        max_files: int = 50
    ) -> List[Tuple[str, str]]:
        """
        Get direct download URLs for all files in a HuggingFace model.
        
        Args:
            model_id: HuggingFace model ID (e.g., "google-bert/bert-base-uncased")
            max_files: Maximum number of files to return
            
        Returns:
            List of (filename, download_url) tuples
        """
        try:
            # List all files in the repository
            files = list_repo_files(model_id, token=self.token)
            
            # Build download URLs
            file_urls = []
            for filename in files[:max_files]:
                # HuggingFace CDN URL pattern
                url = (
                    f"https://huggingface.co/{model_id}"
                    f"/resolve/main/{filename}"
                )
                file_urls.append((filename, url))
            
            logger.info(
                f"Found {len(file_urls)} files for {model_id}"
            )
            return file_urls
            
        except RepositoryNotFoundError:
            logger.warning(f"Repository not found: {model_id}")
            return []
        except Exception as e:
            logger.error(f"Error listing files for {model_id}: {e}")
            return []

    def get_dataset_file_urls(
        self,
        dataset_id: str,
        max_files: int = 50
    ) -> List[Tuple[str, str]]:
        """
        Get direct download URLs for files in a HuggingFace dataset.
        
        Args:
            dataset_id: HuggingFace dataset ID
            max_files: Maximum number of files to return
            
        Returns:
            List of (filename, download_url) tuples
        """
        try:
            files = list_repo_files(
                dataset_id,
                repo_type="dataset",
                token=self.token
            )
            
            file_urls = []
            for filename in files[:max_files]:
                url = (
                    f"https://huggingface.co/datasets/{dataset_id}"
                    f"/resolve/main/{filename}"
                )
                file_urls.append((filename, url))
            
            logger.info(
                f"Found {len(file_urls)} files for dataset {dataset_id}"
            )
            return file_urls
            
        except RepositoryNotFoundError:
            logger.warning(f"Dataset not found: {dataset_id}")
            return []
        except Exception as e:
            logger.error(f"Error listing dataset files for {dataset_id}: {e}")
            return []

    def get_model_info(self, model_url: ParsedURL) -> Optional[Dict[str, Any]]:
        # get comprehensive model information from HF Hub API
        if model_url.platform != "huggingface" or not model_url.repo:
            return None

        try:
            # get basic model info
            repo_id = (
                f"{model_url.owner}/{model_url.repo}"
                if model_url.owner
                else model_url.repo
            )
            
            # Direct synchronous API call
            model_info = self.api.model_info(repo_id, token=self.token)

            # convert to dict and add additional metrics
            info_dict = {
                "id": model_info.id,
                "author": getattr(model_info, "author", None),
                "downloads": getattr(model_info, "downloads", 0),
                "likes": getattr(model_info, "likes", 0),
                "created_at": getattr(model_info, "created_at", None),
                "last_modified": getattr(model_info, "last_modified", None),
                "tags": getattr(model_info, "tags", []),
                "pipeline_tag": getattr(model_info, "pipeline_tag", None),
                "library_name": getattr(model_info, "library_name", None),
                "model_index": getattr(model_info, "model_index", None),
            }
            
            # Extract GitHub repository from card_data if available
            try:
                card_data = getattr(model_info, "card_data", None)
                if card_data:
                    # Check for source code URL in model card metadata
                    source_url = None
                    
                    # Try different common fields for GitHub links
                    has_get = hasattr(card_data, "get")
                    is_callable = callable(getattr(card_data, "get", None))
                    if has_get and is_callable:
                        try:
                            source_url = (
                                card_data.get("source_url") or
                                card_data.get("repo_url") or
                                card_data.get("repository_url") or
                                card_data.get("code_url")
                            )
                        except (AttributeError, TypeError):
                            pass
                    
                    if source_url and "github.com" in source_url:
                        info_dict["github_url"] = source_url
            except Exception:
                # Ignore errors in GitHub extraction
                pass
            
            # Also check tags for GitHub links
            try:
                tags = getattr(model_info, "tags", [])
                if tags:
                    for tag in tags:
                        is_github = (
                            isinstance(tag, str) and
                            "github.com" in tag.lower()
                        )
                        if is_github:
                            info_dict["github_url"] = tag
                            break
            except Exception:
                # Ignore errors in tag processing
                pass

            # Get file information synchronously
            try:
                files = list_repo_files(repo_id, token=self.token)
                info_dict["files"] = files
                info_dict["file_count"] = len(files)
            except Exception as e:
                logger.warning(f"Could not list files for {repo_id}: {e}")
                info_dict["files"] = []
                info_dict["file_count"] = 0

            return info_dict

        except (RepositoryNotFoundError, EntryNotFoundError) as e:
            logger.warning(f"Repository not found: {model_url.url} - {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching model info for {model_url.url}: {e}")
            return None

    def get_dataset_info(
        self, dataset_url: ParsedURL
    ) -> Optional[Dict[str, Any]]:
        # get dataset information from HF Hub API
        if dataset_url.platform != "huggingface" or not dataset_url.repo:
            return None

        try:
            repo_id = (
                f"{dataset_url.owner}/{dataset_url.repo}"
                if dataset_url.owner
                else dataset_url.repo
            )
            
            # Direct synchronous API call
            dataset_info = self.api.dataset_info(repo_id, token=self.token)

            return {
                "id": dataset_info.id,
                "author": getattr(dataset_info, "author", None),
                "downloads": getattr(dataset_info, "downloads", 0),
                "likes": getattr(dataset_info, "likes", 0),
                "created_at": getattr(dataset_info, "created_at", None),
                "last_modified": getattr(dataset_info, "last_modified", None),
                "tags": getattr(dataset_info, "tags", []),
                "task_categories": getattr(dataset_info, "task_categories", []),
            }

        except Exception as e:
            logger.warning(f"Could not fetch dataset info for {dataset_url.url}: {e}")
            return None

    def download_file(
        self, repo_id: str, filename: str, is_dataset: bool = False
    ) -> Optional[str]:
        # download a specific file from a repository
        try:
            # Download file synchronously
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset" if is_dataset else "model",
                token=self.token,
            )

            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return content

        except Exception as e:
            logger.debug(f"Could not download {filename} from {repo_id}: {e}")
            return None

    def get_readme_content(self, parsed_url: ParsedURL) -> Optional[str]:
        # get README content from a repository
        if parsed_url.platform != "huggingface" or not parsed_url.repo:
            return None

        repo_id = (
            f"{parsed_url.owner}/{parsed_url.repo}"
            if parsed_url.owner
            else parsed_url.repo
        )
        is_dataset = parsed_url.category == URLCategory.DATASET

        # try different README file names
        readme_files = ["README.md", "readme.md", "README.txt", "readme.txt"]

        # Try README files sequentially
        for readme_file in readme_files:
            try:
                content = self.download_file(repo_id, readme_file, is_dataset)
                if content:
                    return content
            except Exception:
                continue

        return None

    def get_model_config(self, model_url: ParsedURL) -> Optional[Dict[str, Any]]:
        # get model configuration files
        if model_url.platform != "huggingface" or not model_url.repo:
            return None

        repo_id = (
            f"{model_url.owner}/{model_url.repo}" if model_url.owner else model_url.repo
        )

        config_files = ["config.json", "model_index.json", "tokenizer.json"]

        # Download config files sequentially
        config_data = {}
        for config_file in config_files:
            try:
                content = self.download_file(repo_id, config_file, False)
                if content:
                    try:
                        config_data[config_file] = json.loads(content)
                    except json.JSONDecodeError:
                        logger.debug(f"Could not parse JSON from {config_file}")
            except Exception:
                continue

        return config_data if config_data else None
